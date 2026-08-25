#!/usr/bin/env python3
"""
ECA GUI - Electron Cloud Analysis Graphical Interface
A GUI application for browsing, filtering, fitting, and plotting PyECLOUD simulation data.
"""

import sys
import os
import json
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, filedialog, messagebox, scrolledtext
from typing import Optional, List, Dict, Any, Callable
import numpy as np

try:
    from eca import SimDB, ECModel, WhereIn
    from eca.fit import (
        FurmanNoPhotoFit, FurmanPhotoFit, FurmanNPMCFit, KTYFit, 
        BeforeBunchSelector, BunchAverageSelector
    )
    from eca.plots import model_plot, versus_plot, histogram_plot
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
except ImportError as e:
    print(f"Error importing ECA modules: {e}")
    print("Make sure the 'eca' package is properly installed.")
    sys.exit(1)


# ==================== LOGIC & DATA LAYER ====================

class FilterDefinition:
    """Serializable filter definition for copy-paste support."""
    EXACT, CONDITION, EXPRESSION = "exact", "condition", "expression"
    
    @staticmethod
    def to_dict(filter_obj: Dict[str, Any]) -> Dict[str, Any]:
        result = {"type": filter_obj["type"]}
        if filter_obj["type"] == FilterDefinition.EXACT:
            result["property"] = filter_obj["property"]
            result["values"] = [str(v) if isinstance(v, (list, np.ndarray)) else v for v in filter_obj["values"]]
        elif filter_obj["type"] == FilterDefinition.CONDITION:
            result.update({k: filter_obj[k] for k in ["property", "operator", "value"]})
        else:
            result["expr"] = filter_obj["expr"]
        return result
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Dict[str, Any]:
        return data.copy()
    
    @staticmethod
    def deserialize_all(json_str: str) -> List[Dict[str, Any]]:
        try:
            data = json.loads(json_str)
            if not isinstance(data, list): raise ValueError("Filter data must be a JSON array")
            return [FilterDefinition.from_dict(item) for item in data]
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")


class ECADatabaseHandler:
    """Handles data operations, abstracting logic from the UI."""
    def __init__(self, db_file: Optional[str] = None, db_instance: Optional[SimDB] = None):
        self.db: Optional[SimDB] = db_instance
        self.search_criteria: Dict[str, Any] = {}
        if db_file:
            self.load_database(db_file)

    def load_database(self, filename: str) -> int:
        self.db = SimDB(filename, verbose=True)
        sims = self.db.where()
        if sims and sims[0].get('path', '').startswith('.'):
            os.chdir(os.path.dirname(os.path.abspath(filename)))
        self.search_criteria = {}
        return len(self.db.db)

    def save_database(self, filename: str):
        if not self.db: return
        from tinydb import TinyDB
        from tinydb.storages import JSONStorage
        from tinydb.middlewares import CachingMiddleware
        
        new_db = TinyDB(filename, storage=CachingMiddleware(JSONStorage))
        new_db.insert_multiple(self.db.db.all())
        new_db.close()

    def get_filtered_sims(self) -> List[Dict]:
        if not self.db: return []
        return self.db.where(**self.search_criteria)

    def get_all_keys(self) -> List[str]:
        return self.db.all_keys() if self.db else []

    def get_unique_values(self, key: str) -> List[Any]:
        return self.db.unique(key, **self.search_criteria) if self.db else []

    def apply_filters(self, filters_json: str):
        self.search_criteria.clear()
        filters = FilterDefinition.deserialize_all(filters_json)
        
        for i, f_data in enumerate(filters):
            if f_data["type"] == FilterDefinition.EXACT:
                self.search_criteria[f_data["property"]] = WhereIn(*f_data["values"])
            elif f_data["type"] == FilterDefinition.CONDITION:
                self.search_criteria[f"_cond_{i}"] = self._make_condition(f_data["property"], f_data["operator"], f_data["value"])
            else:
                self.search_criteria[f"_expr_{i}"] = f_data["expr"]

    @staticmethod
    def _make_condition(prop: str, op: str, val: Any) -> Callable:
        def cond_fn(result):
            if prop not in result: return False
            res_val = result[prop]
            try:
                ops = {">": res_val > val, "<": res_val < val, ">=": res_val >= val, 
                       "<=": res_val <= val, "==": res_val == val, "!=": res_val != val}
                return ops.get(op, False)
            except TypeError:
                return False
        return cond_fn


# ==================== UI LAYER ====================

class ECAApp:
    """Main application class for the GUI."""
    
    FIT_MODELS = {
        "FurmanNoPhoto": FurmanNoPhotoFit,
        "FurmanNPMC": FurmanNPMCFit,
        "FurmanPhoto": FurmanPhotoFit,
        "KTY": KTYFit
    }
    
    def __init__(self, root: tk.Tk, handler: Optional[ECADatabaseHandler] = None, is_temp: bool = False):
        self.root = root
        self.is_temp = is_temp
        self.handler = handler or ECADatabaseHandler()
        
        self.filter_val_map = {}
        self.individual_columns = ["doc_id"]
        self._last_doc_id = None
        self._last_fits = None
        
        self._setup_window()
        self._setup_fonts()
        self._setup_ui()
        
        if self.handler.db:
            self._on_db_loaded()

    def _setup_window(self):
        title_prefix = "[TEMP WINDOW] " if self.is_temp else ""
        self.root.title(f"{title_prefix}ECA GUI - Electron Cloud Analysis")
        self.root.geometry("1400x900")
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _setup_fonts(self):
        for font_name in ["TkDefaultFont", "TkTextFont", "TkFixedFont"]:
            tkfont.nametofont(font_name).configure(size=11)
        style = ttk.Style()
        style.configure(".", font="TkDefaultFont")
        style.configure("Treeview.Heading", font="TkDefaultFont", weight="bold")

    def _on_closing(self):
        if messagebox.askokcancel("Quit", f"Do you want to close this {'temporary ' if self.is_temp else ''}window?"):
            self.root.destroy()
            if not self.is_temp: sys.exit(0)

    def _update_status(self, message: str):
        self.status_var.set(message)
        self.root.update_idletasks()

    @staticmethod
    def _format_value(val: Any) -> str:
        if isinstance(val, bool): return str(val)
        if isinstance(val, (int, float, np.number)):
            if np.isnan(val) or np.isinf(val): return str(val)
            abs_val = abs(val)
            if abs_val >= 1000 or abs_val < 1e-4: return f"{val:.3E}"
            if isinstance(val, (float, np.floating)): return f"{val:.4f}"
            return str(int(val))
        return str(val)

    # --- UI Builders ---

    def _make_combo(self, parent, label_text, var, values=None, width=20, side=tk.LEFT):
        frame = ttk.Frame(parent)
        frame.pack(side=side, padx=5, fill=tk.X)
        if label_text: ttk.Label(frame, text=label_text).pack(side=tk.LEFT)
        combo = ttk.Combobox(frame, textvariable=var, state="readonly", width=width, values=values or [])
        combo.pack(side=tk.LEFT, padx=5)
        return combo

    def _make_treeview(self, parent, columns, heights=20):
        tree = ttk.Treeview(parent, columns=columns, show="headings", height=heights)
        for col in columns:
            tree.heading(col, text=col, command=lambda _col=col: self._sort_treeview(tree, _col, False))
        scroll = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        return tree

    def _sort_treeview(self, tree, col, reverse):
        """Sort treeview content when a column header is clicked."""
        l = [(tree.set(k, col), k) for k in tree.get_children('')]
        try: l.sort(key=lambda t: float(t[0]), reverse=reverse)
        except ValueError: l.sort(reverse=reverse)
        for index, (val, k) in enumerate(l): tree.move(k, '', index)
        tree.heading(col, command=lambda: self._sort_treeview(tree, col, not reverse))

    def _setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1); self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1); main_frame.rowconfigure(1, weight=1)
        
        self._create_toolbar(main_frame)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=1, column=0, sticky="nsew", pady=5)
        
        self._create_overview_tab()
        self._create_filter_tab()
        self._create_fitting_tab()
        self._create_plotting_tab()
        self._create_histogram_tab()
        self._create_individual_plot_tab()
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).grid(row=2, column=0, sticky="ew")

    def _create_toolbar(self, parent):
        toolbar = ttk.Frame(parent)
        toolbar.grid(row=0, column=0, sticky="ew", pady=5)
        
        ttk.Button(toolbar, text="Load Database", command=self._load_database).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Clear Database", command=self._clear_database).pack(side=tk.LEFT, padx=2)
        if self.is_temp: ttk.Button(toolbar, text="Save As...", command=self._save_database).pack(side=tk.LEFT, padx=2)
        
        self.info_label = ttk.Label(toolbar, text="No database loaded", font=("TkDefaultFont", 11, "bold"))
        self.info_label.pack(side=tk.RIGHT, padx=5)

    # --- Actions ---

    def _load_database(self):
        filename = filedialog.askopenfilename(filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if not filename: return
        try:
            self._update_status(f"Loading {filename}...")
            count = self.handler.load_database(filename)
            self._on_db_loaded()
            self.info_label.config(text=f"Loaded: {os.path.basename(filename)}")
            self._update_status(f"Loaded {count} simulations")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load: {e}")

    def _on_db_loaded(self):
        self.individual_columns = ["doc_id"]
        self._refresh_all_tabs()

    def _save_database(self):
        filename = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if filename:
            try:
                self.handler.save_database(filename)
                messagebox.showinfo("Success", "Saved successfully!")
                self.info_label.config(text=f"Loaded: {os.path.basename(filename)}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def _clear_database(self):
        if messagebox.askyesno("Confirm", "Clear the database?"):
            self.handler = ECADatabaseHandler()
            self._clear_all_tabs()
            self.info_label.config(text="No database loaded")
            self._update_status("Database cleared")

    def _clear_all_tabs(self):
        for tree in [self.sim_tree, self.individual_sim_tree]:
            tree.delete(*tree.get_children())
        for txt in [self.prop_text, self.active_filters_text, self.results_text]:
            txt.delete(1.0, tk.END)
        for fig, canvas in [(self.plot_fig, self.plot_canvas), (self.hist_fig, self.hist_canvas), (self.individual_fig, self.individual_canvas)]:
            fig.clear(); canvas.draw()

    def _refresh_all_tabs(self):
        self._update_overview_tab()
        self._populate_dropdowns()
        self._update_individual_sim_list()

    # --- Overview Tab ---

    def _create_overview_tab(self):
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="Overview")
        tab.columnconfigure(0, weight=1); tab.columnconfigure(1, weight=1); tab.rowconfigure(0, weight=1)
        
        l_frame = ttk.LabelFrame(tab, text="Simulations", padding="5")
        l_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.sim_tree = self._make_treeview(l_frame, ["Path"])
        self.sim_tree.column("Path", width=500)
        
        r_frame = ttk.LabelFrame(tab, text="Properties Summary", padding="5")
        r_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        self.prop_text = scrolledtext.ScrolledText(r_frame, wrap=tk.WORD)
        self.prop_text.pack(fill=tk.BOTH, expand=True)

    def _update_overview_tab(self):
        if not self.handler.db: return
        self.sim_tree.delete(*self.sim_tree.get_children())
        self.prop_text.delete(1.0, tk.END)
        
        sims = self.handler.get_filtered_sims()
        for sim in sims:
            path = sim.get('path', 'Unknown')
            self.sim_tree.insert("", tk.END, values=("(Path not found) " + path if not os.path.exists(path) else path,))
        
        property_stats = {}
        for key in self.handler.get_all_keys():
            if key == 'path': continue
            vals = self.handler.get_unique_values(key)
            if 1 < len(vals) < len(sims):
                property_stats[key] = vals

        self.prop_text.insert(tk.END, f"Total Displayed: {len(sims)}\n{'='*40}\nProperties with multiple values ({len(property_stats)}):\n\n")
        for prop, vals in sorted(property_stats.items()):
            self.prop_text.insert(tk.END, f"{prop}: {len(vals)} unique values\n  Values: {', '.join(self._format_value(v) for v in vals[:10])}{' ...' if len(vals) > 10 else ''}\n\n")

    # --- Filter Tab ---

    def _create_filter_tab(self):
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="Filter")
        tab.columnconfigure(0, weight=1); tab.rowconfigure(2, weight=1)
        
        # Mode
        mode_frame = ttk.Frame(tab)
        mode_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(mode_frame, text="Filter Mode:").pack(side=tk.LEFT)
        self.filter_mode_var = tk.StringVar(value=FilterDefinition.EXACT)
        for mode, lbl in [(FilterDefinition.EXACT, "Exact Match"), (FilterDefinition.CONDITION, "Condition"), (FilterDefinition.EXPRESSION, "Custom Expression")]:
            ttk.Radiobutton(mode_frame, text=lbl, variable=self.filter_mode_var, value=mode, command=self._toggle_filter_mode).pack(side=tk.LEFT, padx=5)
        
        # Inputs
        control_frame = ttk.LabelFrame(tab, text="Filter Criteria", padding="5")
        control_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        
        self.filter_property_var = tk.StringVar()
        self.prop_frame = ttk.Frame(control_frame)
        self.filter_property_combo = self._make_combo(self.prop_frame, "Property:", self.filter_property_var, width=30)
        self.filter_property_combo.bind("<<ComboboxSelected>>", self._on_property_select)
        
        self.input_container = ttk.Frame(control_frame)
        self.input_container.pack(fill=tk.X, pady=5)
        
        # Exact Frame
        self.exact_frame = ttk.Frame(self.input_container)
        ttk.Label(self.exact_frame, text="Values:").pack(side=tk.LEFT)
        self.filter_values_listbox = tk.Listbox(self.exact_frame, selectmode=tk.MULTIPLE, height=5, width=50)
        self.filter_values_listbox.pack(side=tk.LEFT, padx=5)
        
        # Condition Frame
        self.condition_frame = ttk.Frame(self.input_container)
        self.cond_op_var = tk.StringVar(value=">")
        self._make_combo(self.condition_frame, "Operator:", self.cond_op_var, values=[">", "<", ">=", "<=", "==", "!="], width=5)
        self.cond_val_var = tk.StringVar()
        ttk.Label(self.condition_frame, text="Value:").pack(side=tk.LEFT)
        ttk.Entry(self.condition_frame, textvariable=self.cond_val_var, width=20).pack(side=tk.LEFT, padx=5)
        
        # Expression Frame
        self.expression_frame = ttk.Frame(self.input_container)
        self.expr_var = tk.StringVar()
        ttk.Label(self.expression_frame, text="Expr:").pack(side=tk.LEFT)
        ttk.Entry(self.expression_frame, textvariable=self.expr_var, width=50).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Add Filter", command=self._add_filter).pack(pady=5)
        
        # Output
        filters_frame = ttk.LabelFrame(tab, text="Active Filters (JSON)", padding="5")
        filters_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 10))
        self.active_filters_text = scrolledtext.ScrolledText(filters_frame, height=8, font=("TkFixedFont", 9))
        self.active_filters_text.pack(fill=tk.BOTH, expand=True)
        
        btn_frame = ttk.Frame(tab)
        btn_frame.grid(row=3, column=0, pady=10)
        for txt, cmd in [("Apply", self._apply_filter), ("Clear", self._clear_filters), ("Reset DB", self._reset_filter), ("New Temp Window", self._open_temp_window)]:
            ttk.Button(btn_frame, text=txt, command=cmd).pack(side=tk.LEFT, padx=5)
            
        self._toggle_filter_mode()

    def _toggle_filter_mode(self):
        for f in [self.exact_frame, self.condition_frame, self.expression_frame, self.prop_frame]: f.pack_forget()
        mode = self.filter_mode_var.get()
        if mode != FilterDefinition.EXPRESSION: self.prop_frame.pack(fill=tk.X, pady=5)
        {"exact": self.exact_frame, "condition": self.condition_frame, "expression": self.expression_frame}[mode].pack(fill=tk.X)

    def _on_property_select(self, e=None):
        prop = self.filter_property_var.get()
        if not prop or not self.handler.db: return
        vals = set(tuple(v) if isinstance(v, list) else v for doc in self.handler.db.db.all() if prop in doc for v in [doc[prop]])
        self.filter_val_map = {self._format_value(v): v for v in sorted(list(vals), key=str)}
        self.filter_values_listbox.delete(0, tk.END)
        self.filter_values_listbox.insert(tk.END, *self.filter_val_map.keys())

    def _add_filter(self):
        mode = self.filter_mode_var.get()
        try:
            if mode == FilterDefinition.EXACT:
                sel = [self.filter_values_listbox.get(i) for i in self.filter_values_listbox.curselection()]
                if not sel: raise ValueError("Select at least one value")
                f = {"type": mode, "property": self.filter_property_var.get(), "values": [self.filter_val_map[s] for s in sel]}
            elif mode == FilterDefinition.CONDITION:
                val = self.cond_val_var.get().strip()
                try: val = int(val) if float(val).is_integer() else float(val)
                except ValueError: pass
                f = {"type": mode, "property": self.filter_property_var.get(), "operator": self.cond_op_var.get(), "value": val}
            else:
                f = {"type": mode, "expr": self.expr_var.get().strip()}
            
            txt = self.active_filters_text.get(1.0, tk.END).strip()
            filters = json.loads(txt) if txt and txt != "[]" else []
            filters.append(f)
            self.active_filters_text.delete(1.0, tk.END); self.active_filters_text.insert(tk.END, json.dumps(filters, indent=2))
        except Exception as e: messagebox.showwarning("Warning", str(e))

    def _apply_filter(self):
        try:
            self.handler.apply_filters(self.active_filters_text.get(1.0, tk.END).strip() or "[]")
            self._update_overview_tab(); self._update_individual_sim_list()
            messagebox.showinfo("Success", f"Filter applied. {len(self.handler.get_filtered_sims())} match.")
        except Exception as e: messagebox.showerror("Error", str(e))

    def _clear_filters(self):
        self.active_filters_text.delete(1.0, tk.END); self.active_filters_text.insert(tk.END, "[]")

    def _reset_filter(self):
        self._clear_filters(); self._apply_filter()

    def _open_temp_window(self):
        if not self.handler.search_criteria: return messagebox.showwarning("Warning", "Apply filters first.")
        new_app = ECAApp(tk.Toplevel(self.root), handler=self.handler, is_temp=True)
        new_app.handler.search_criteria = self.handler.search_criteria.copy()
        new_app._on_db_loaded()

    # --- Fitting Tab ---

    def _create_fitting_tab(self):
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="Fitting")
        tab.columnconfigure(0, weight=1); tab.rowconfigure(5, weight=1)
        
        m_frame = ttk.LabelFrame(tab, text="Fit Model", padding="5")
        m_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.fit_model_var = tk.StringVar(value="FurmanNoPhoto")
        self._make_combo(m_frame, "Model:", self.fit_model_var, values=list(self.FIT_MODELS.keys()), width=30)
        
        s_frame = ttk.LabelFrame(tab, text="Data Selection", padding="5")
        s_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self.fit_selector_var = tk.StringVar(value="BunchAverage")
        self._make_combo(s_frame, "Selector:", self.fit_selector_var, values=["BunchAverage", "BeforeBunch"], width=15)
        
        self.fit_central_density_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(s_frame, text="Use Central Density", variable=self.fit_central_density_var).pack(side=tk.LEFT, padx=10)
        self.fit_train_var = tk.StringVar(value="-1")
        ttk.Label(s_frame, text="Train:").pack(side=tk.LEFT); ttk.Entry(s_frame, textvariable=self.fit_train_var, width=5).pack(side=tk.LEFT)
        
        sel_frame = ttk.LabelFrame(tab, text="Selection", padding="5")
        sel_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 10))
        self.selection_mode = tk.StringVar(value="filtered")
        ttk.Radiobutton(sel_frame, text="Filtered Simulations", variable=self.selection_mode, value="filtered").pack(anchor=tk.W)
        ttk.Radiobutton(sel_frame, text="All Simulations", variable=self.selection_mode, value="all").pack(anchor=tk.W)
        
        self.progress_var = tk.StringVar(value="Ready")
        ttk.LabelFrame(tab, text="Progress").grid(row=3, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(tab.grid_slaves(row=3, column=0)[0], textvariable=self.progress_var, padding=5).pack(fill=tk.X)
        
        btn_frame = ttk.Frame(tab)
        btn_frame.grid(row=4, column=0, pady=10)
        ttk.Button(btn_frame, text="Apply Fit", command=self._apply_fit).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Refit All", command=lambda: self._apply_fit(refit=True)).pack(side=tk.LEFT, padx=5)
        
        res_frame = ttk.LabelFrame(tab, text="Fitting Results", padding="5")
        res_frame.grid(row=5, column=0, sticky="nsew")
        self.results_text = scrolledtext.ScrolledText(res_frame, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True)

    def _apply_fit(self, refit: bool = False):
        if not self.handler.db: return messagebox.showwarning("Warning", "No database")
        
        docs = [doc for doc in (self.handler.get_filtered_sims() if self.selection_mode.get() == "filtered" else self.handler.db.where())]
        if not docs: return messagebox.showwarning("Warning", "No sims to fit")
        
        m_name = self.fit_model_var.get()
        sel_class = BeforeBunchSelector if self.fit_selector_var.get() == "BeforeBunch" else BunchAverageSelector
        try: train = int(self.fit_train_var.get())
        except ValueError: return messagebox.showwarning("Warning", "Train must be an int")
        
        selector = sel_class(use_central_density=self.fit_central_density_var.get(), use_train=train)
        fit_class = self.FIT_MODELS[m_name]
        fit = fit_class(self.handler.db, selector=selector) if m_name == "FurmanNPMC" else fit_class(selector=selector)
        
        self.progress_var.set("Starting fit...")
        success = 0
        for i, doc in enumerate(docs):
            model = ECModel(self.handler.db.db, doc)
            try: result = fit.fit(model, refit=refit)
            except: result = None
            if result is not None: success += 1
            self.progress_var.set(f"Fitting: {i+1}/{len(docs)}")
            self.root.update()
                
        self.results_text.delete(1.0, tk.END); self.results_text.insert(tk.END, f"Done.\nSuccessful: {success}/{len(docs)}\n")

    # --- Plotting & Histograms Tabs ---

    def _make_plot_tab(self, name, is_hist=False):
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text=name)
        tab.columnconfigure(0, weight=1); tab.rowconfigure(1, weight=1)
        
        ctrl = ttk.LabelFrame(tab, text="Controls", padding="5")
        ctrl.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        if is_hist:
            self.hist_prop_var = tk.StringVar()
            self.hist_prop_combo = self._make_combo(ctrl, "Prop:", self.hist_prop_var)
            self.hist_bins_var = tk.StringVar(value="auto")
            ttk.Label(ctrl, text="Bins:").pack(side=tk.LEFT); ttk.Entry(ctrl, textvariable=self.hist_bins_var, width=10).pack(side=tk.LEFT)
            self.hist_log_var = tk.BooleanVar()
            ttk.Checkbutton(ctrl, text="Log Y", variable=self.hist_log_var).pack(side=tk.LEFT, padx=10)
            ttk.Button(ctrl, text="Plot", command=self._plot_hist).pack(side=tk.LEFT, padx=10)
            fig, self.hist_fig = Figure(figsize=(8,6), dpi=100), Figure(figsize=(8,6), dpi=100)
            canvas = FigureCanvasTkAgg(self.hist_fig, master=ttk.LabelFrame(tab, text="Plot").grid(row=1, column=0, sticky="nsew") or tab.grid_slaves(row=1)[0])
            self.hist_canvas = canvas
        else:
            self.plot_x_var, self.plot_y_var, self.plot_color_var = tk.StringVar(), tk.StringVar(), tk.StringVar(value="None")
            self.plot_x_combo = self._make_combo(ctrl, "X:", self.plot_x_var)
            self.plot_y_combo = self._make_combo(ctrl, "Y:", self.plot_y_var)
            self.plot_color_combo = self._make_combo(ctrl, "Color:", self.plot_color_var)
            ttk.Button(ctrl, text="Plot", command=self._plot_scatter).pack(side=tk.LEFT, padx=10)
            self.plot_fig = Figure(figsize=(8,6), dpi=100)
            canvas = FigureCanvasTkAgg(self.plot_fig, master=ttk.LabelFrame(tab, text="Plot").grid(row=1, column=0, sticky="nsew") or tab.grid_slaves(row=1)[0])
            self.plot_canvas = canvas
            
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas, tab.grid_slaves(row=1)[0]).update()

    def _create_plotting_tab(self): self._make_plot_tab("Scatter Plots", False)
    def _create_histogram_tab(self): self._make_plot_tab("Histograms", True)

    def _populate_dropdowns(self):
        if not self.handler.db: return
        
        # All available keys except 'path'
        all_keys = [k for k in self.handler.get_all_keys() if k != 'path']
        
        # For filters: properties with more than 1 unique value are useful to filter on
        filter_props = [k for k in all_keys if 1 < len(self.handler.get_unique_values(k))]
        
        # For plots: strictly properties that contain numeric data
        plot_props = [p for p in filter_props if any(isinstance(v, (int, float, np.number)) for v in self.handler.get_unique_values(p))]
        
        # Assign to the UI comboboxes
        self.filter_property_combo['values'] = filter_props
        self.plot_x_combo['values'] = self.plot_y_combo['values'] = self.hist_prop_combo['values'] = plot_props
        self.plot_color_combo['values'] = ["None"] + plot_props
        self.indiv_col_combo['values'] = all_keys

    def _plot_scatter(self):
        if not (self.plot_x_var.get() and self.plot_y_var.get()): return
        self.plot_fig.clf()
        db = self.handler.db.extract(self.handler.get_all_keys(), **self.handler.search_criteria) if self.handler.search_criteria else self.handler.db
        versus_plot(db, self.plot_x_var.get(), self.plot_y_var.get(), colorBy=self.plot_color_var.get() if self.plot_color_var.get() != "None" else None, size=self.plot_fig.add_subplot(111))
        self.plot_canvas.draw()

    def _plot_hist(self):
        if not self.hist_prop_var.get(): return
        b = self.hist_bins_var.get().strip()
        b = int(b) if b.isdigit() else 'auto'
        self.hist_fig.clf()
        db = self.handler.db.extract(self.handler.get_all_keys(), **self.handler.search_criteria) if self.handler.search_criteria else self.handler.db
        histogram_plot(db, self.hist_prop_var.get(), bins=b, log_y=self.hist_log_var.get(), size=self.hist_fig.add_subplot(111))
        self.hist_canvas.draw()

    # --- Individual Plot Tab ---

    def _create_individual_plot_tab(self):
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="Individual Plot")
        tab.columnconfigure(1, weight=1); tab.rowconfigure(0, weight=1)
        
        l_frame = ttk.LabelFrame(tab, text="Select Simulation", padding="5")
        l_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        c_frame = ttk.Frame(l_frame)
        c_frame.pack(fill=tk.X, pady=5)
        self.indiv_col_var = tk.StringVar()
        self.indiv_col_combo = self._make_combo(c_frame, "Add Col:", self.indiv_col_var, width=15)
        ttk.Button(c_frame, text="Add", command=lambda: self._mod_cols(add=True)).pack(side=tk.LEFT, padx=2)
        ttk.Button(c_frame, text="Reset", command=lambda: self._mod_cols(add=False)).pack(side=tk.LEFT)
        
        self.individual_sim_tree = self._make_treeview(l_frame, self.individual_columns, heights=15)
        self.individual_sim_tree.bind("<<TreeviewSelect>>", lambda e: self._plot_indiv())
        
        r_frame = ttk.Frame(tab)
        r_frame.grid(row=0, column=1, sticky="nsew")
        r_frame.columnconfigure(0, weight=1); r_frame.rowconfigure(1, weight=1)
        
        fit_frame = ttk.LabelFrame(r_frame, text="Configuration", padding="5")
        fit_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        self.individual_fit_vars = {name: tk.BooleanVar() for name in self.FIT_MODELS}
        for name, var in self.individual_fit_vars.items():
            ttk.Checkbutton(fit_frame, text=name, variable=var, command=self._plot_indiv).pack(side=tk.LEFT, padx=2)
            
        self.individual_central_density_var = tk.BooleanVar()
        ttk.Checkbutton(fit_frame, text="Use CD", variable=self.individual_central_density_var, command=self._plot_indiv).pack(side=tk.LEFT, padx=5)
        
        self.individual_train_var, self.individual_max_x_var = tk.StringVar(value="All"), tk.StringVar()
        self.individual_train_combo = self._make_combo(fit_frame, "Train:", self.individual_train_var, width=5)
        self.individual_train_combo.bind("<<ComboboxSelected>>", lambda e: self._plot_indiv())
        ttk.Label(fit_frame, text="Max X:").pack(side=tk.LEFT); ttk.Entry(fit_frame, textvariable=self.individual_max_x_var, width=8).pack(side=tk.LEFT)
        ttk.Button(fit_frame, text="Replot", command=self._plot_indiv).pack(side=tk.LEFT, padx=5)
        
        plot_frame = ttk.LabelFrame(r_frame, text="Plot", padding="5")
        plot_frame.grid(row=1, column=0, sticky="nsew")
        self.individual_fig = Figure(figsize=(10, 6), dpi=100)
        self.individual_canvas = FigureCanvasTkAgg(self.individual_fig, master=plot_frame)
        self.individual_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.individual_canvas, plot_frame).update()

    def _mod_cols(self, add: bool):
        col = self.indiv_col_var.get()
        if add and col and col not in self.individual_columns: self.individual_columns.append(col)
        elif not add: self.individual_columns = ["doc_id"]
        self._update_individual_sim_list()

    def _update_individual_sim_list(self):
        if not self.handler.db: return
        self.individual_sim_tree.delete(*self.individual_sim_tree.get_children())
        self.individual_sim_tree.configure(columns=self.individual_columns)
        for col in self.individual_columns:
            self.individual_sim_tree.heading(col, text=col, command=lambda _col=col: self._sort_treeview(self.individual_sim_tree, _col, False))
            self.individual_sim_tree.column(col, width=60 if col == "doc_id" else 120)
        
        for sim in self.handler.get_filtered_sims():
            self.individual_sim_tree.insert("", tk.END, values=[sim.doc_id if c == "doc_id" else self._format_value(sim.get(c, "N/A")) for c in self.individual_columns])

    def _plot_indiv(self):
        selections = self.individual_sim_tree.selection()
        if not selections or not self.handler.db: return
        
        self.individual_fig.clf()
        ax = self.individual_fig.add_subplot(111)
        sel_fits = [n for n, v in self.individual_fit_vars.items() if v.get()]
        
        # Restore the correct train parameter expectation ("All" vs int) for model_plot
        t_val = self.individual_train_var.get().strip()
        if not t_val or t_val.lower() == "all":
            plot_train = "All"
        else:
            try:
                plot_train = int(t_val)
            except ValueError:
                plot_train = "All"

        mx_val = self.individual_max_x_var.get().strip()
        plotted = 0

        for sel in selections:
            try:
                # Handle single-column scalar vs multi-column tuple variations safely
                item_values = self.individual_sim_tree.item(sel)['values']
                doc_id = int(item_values[0]) if isinstance(item_values, (list, tuple)) else int(item_values)
                
                model = ECModel(self.handler.db.db, doc_id)
                
                # Dynamically update the train dropdown configuration on single selection
                if len(selections) == 1 and hasattr(model, 'train_times'):
                    self.individual_train_combo['values'] = ["All"] + [str(i) for i in range(len(model.train_times))]
                
                fits = [self.FIT_MODELS[n](self.handler.db) if n == "FurmanNPMC" else self.FIT_MODELS[n]() for n in sel_fits]
                
                mx = float(mx_val) if mx_val else (model.cutoff / model.bunch_step) * 1.25
                if len(selections) == 1 and not mx_val:
                    self.individual_max_x_var.set(f"{mx:.1f}")
                
                model_plot(
                    model=model, 
                    fits=fits, 
                    size=ax, 
                    show_error=True, 
                    central_density=self.individual_central_density_var.get(), 
                    fit_maxX=mx, 
                    label=str(model.doc_id), 
                    train=plot_train
                )
                plotted += 1
            except Exception as e:
                print(f"Error rendering individual plot for simulation item {sel}: {e}", file=sys.stderr)
                continue
                
        if plotted > 0:
            self.individual_fig.tight_layout()
            self.individual_canvas.draw()
            self._update_status(f"Plotted {plotted} simulation(s)")

def main():
    root = tk.Tk()
    db_file = sys.argv[1] if len(sys.argv) > 1 else None
    app = ECAApp(root, handler=ECADatabaseHandler(db_file=db_file))
    root.mainloop()

if __name__ == "__main__":
    main()