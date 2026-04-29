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
from collections import defaultdict

try:
    from eca import SimDB, ECModel, InstabilityModel, FurmanNoPhotoFit, FurmanPhotoFit, FurmanNPMCFit, Fit, WhereIn, BeforeBunchSelector, BunchAverageSelector
    from ecaplots import model_plot, versus_plot, histogram_plot
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
except ImportError as e:
    print(f"Error importing ECA modules: {e}")
    print("Make sure eca.py and ecaplots.py are in the same directory or PYTHONPATH")
    sys.exit(1)


class FilterDefinition:
    """Serializable filter definition for copy-paste support."""
    
    @staticmethod
    def to_dict(filter_obj: Dict[str, Any]) -> Dict[str, Any]:
        """Convert filter object to serializable dictionary."""
        result = {"type": filter_obj["type"]}
        
        if filter_obj["type"] == "exact":
            result["property"] = filter_obj["property"]
            result["values"] = [
                str(v) if isinstance(v, (list, np.ndarray)) else v 
                for v in filter_obj["values"]
            ]
        elif filter_obj["type"] == "condition":
            result["property"] = filter_obj["property"]
            result["operator"] = filter_obj["operator"]
            result["value"] = filter_obj["value"]
        else:  # expression
            result["expr"] = filter_obj["expr"]
        
        return result
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Dict[str, Any]:
        """Reconstruct filter object from serialized dictionary."""
        if data["type"] == "exact":
            return {
                "type": "exact",
                "property": data["property"],
                "values": data["values"]
            }
        elif data["type"] == "condition":
            return {
                "type": "condition",
                "property": data["property"],
                "operator": data["operator"],
                "value": data["value"]
            }
        else:  # expression
            return {
                "type": "expression",
                "expr": data["expr"]
            }
    
    @staticmethod
    def serialize_all(filters: List[Dict[str, Any]]) -> str:
        """Serialize all filters to JSON string."""
        return json.dumps([FilterDefinition.to_dict(f) for f in filters], indent=2)
    
    @staticmethod
    def deserialize_all(json_str: str) -> List[Dict[str, Any]]:
        """Deserialize filters from JSON string."""
        try:
            data = json.loads(json_str)
            if not isinstance(data, list):
                raise ValueError("Filter data must be a JSON array")
            return [FilterDefinition.from_dict(item) for item in data]
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")


class ECAApp:
    """Main application class for the Electron Cloud Analysis GUI."""
    
    FIT_MODELS = {"FurmanNoPhoto": FurmanNoPhotoFit, "FurmanNPMC": FurmanNPMCFit, "FurmanPhoto": FurmanPhotoFit}
    
    def __init__(self, root: tk.Tk, db: Optional[SimDB] = None, is_temp: bool = False):
        self.root = root
        self.is_temp = is_temp
        self.db: Optional[SimDB] = db
        self.search_criteria: Dict[str, Any] = {}
        self.current_models: List[ECModel] = []
        self.active_filters_list: List[Dict[str, Any]] = []
        self.filter_val_map = {}
        self.individual_columns = ["doc_id"]
        
        self._setup_window()
        self._setup_fonts()
        self._setup_ui()
        self._load_db_if_available()

    def _setup_window(self):
        """Initialize window properties."""
        title_prefix = "[TEMP WINDOW] " if self.is_temp else ""
        self.root.title(f"{title_prefix}ECA GUI - Electron Cloud Analysis")
        self.root.geometry("1400x900")
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _setup_fonts(self):
        """Configure application fonts."""
        for font_name in ["TkDefaultFont", "TkTextFont", "TkFixedFont"]:
            tkfont.nametofont(font_name).configure(size=11)
        
        style = ttk.Style()
        style.configure(".", font="TkDefaultFont")
        style.configure("Treeview.Heading", font="TkDefaultFont", weight="bold")

    def _load_db_if_available(self):
        """Populate UI if initialized with an existing database."""
        if not self.db:
            return
        
        self._update_overview_tab()
        self._populate_filter_options()
        self._populate_plot_options()
        self._populate_individual_options()
        self._update_individual_sim_list()
        
        label_text = "Loaded: Extracted Temp DB" if self.is_temp else "Loaded Database"
        self.info_label.config(text=label_text)
        self._update_status(f"Loaded {len(self.db.where())} simulations")

    def _on_closing(self):
        """Handle window closing."""
        window_type = "temporary " if self.is_temp else ""
        if messagebox.askokcancel("Quit", f"Do you want to close this {window_type}window?"):
            self.root.destroy()
            if not self.is_temp:
                sys.exit(0)

    def _update_status(self, message: str):
        """Update the status bar."""
        self.status_var.set(message)
        self.root.update_idletasks()

    def _format_value(self, val: Any) -> str:
        """Format values consistently for display."""
        if isinstance(val, bool):
            return str(val)
        
        if isinstance(val, (int, float, np.number)):
            if np.isnan(val) or np.isinf(val):
                return str(val)
            
            abs_val = abs(val)
            if abs_val >= 1000 or abs_val < 1e-4:
                return f"{val:.3E}"
            elif isinstance(val, (float, np.floating)):
                return f"{val:.4f}"
            else:
                return str(int(val))
        
        return str(val)

    # ==================== UI SETUP ====================

    def _setup_ui(self):
        """Setup the main user interface."""
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
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
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=2, column=0, sticky="ew")

    def _create_toolbar(self, parent):
        """Create the top toolbar with file operations."""
        toolbar = ttk.Frame(parent)
        toolbar.grid(row=0, column=0, sticky="ew", pady=5)
        
        ttk.Button(toolbar, text="Load Database", command=self._load_database).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Clear Database", command=self._clear_database).pack(side=tk.LEFT, padx=2)
        
        if self.is_temp:
            ttk.Button(toolbar, text="Save Database As...", command=self._save_database).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(toolbar, text="").pack(side=tk.LEFT, expand=True)
        self.info_label = ttk.Label(toolbar, text="No database loaded", font=("TkDefaultFont", 11, "bold"))
        self.info_label.pack(side=tk.RIGHT, padx=5)

    # ==================== DATABASE OPERATIONS ====================

    def _load_database(self):
        """Open file dialog and load a database."""
        filename = filedialog.askopenfilename(
            title="Select Database File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            self._update_status(f"Loading database from {filename}...")
            self.root.update()
            
            self.db = SimDB(filename, verbose=True)
            simulations = self.db.where()
            
            if simulations and simulations[0].get('path', '').startswith('.'):
                db_dir = os.path.dirname(os.path.abspath(filename))
                os.chdir(db_dir)
            
            self._reset_app_state()
            self._refresh_all_tabs()
            
            self.info_label.config(text=f"Loaded: {os.path.basename(filename)}")
            self._update_status(f"Loaded {len(self.db.db)} simulations")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load database: {str(e)}")
            self._update_status("Error loading database")

    def _reset_app_state(self):
        """Reset all application state variables."""
        self.search_criteria = {}
        self.active_filters_list = []
        self.current_models = []
        self.individual_columns = ["doc_id"]

    def _refresh_all_tabs(self):
        """Refresh all tab displays."""
        self._update_overview_tab()
        self._populate_filter_options()
        self._populate_plot_options()
        self._populate_individual_options()
        self._update_individual_sim_list()

    def _save_database(self):
        """Save the in-memory database to a physical JSON file."""
        if not self.db:
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save Database As",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            from tinydb import TinyDB
            from tinydb.storages import JSONStorage
            from tinydb.middlewares import CachingMiddleware
            
            new_db = TinyDB(filename, storage=CachingMiddleware(JSONStorage))
            new_db.insert_multiple(self.db.db.all())
            new_db.close()
            
            messagebox.showinfo("Success", f"Database successfully saved to:\n{filename}")
            self.info_label.config(text=f"Loaded: {os.path.basename(filename)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save database: {str(e)}")

    def _clear_database(self):
        """Clear the current database."""
        if messagebox.askyesno("Confirm", "Are you sure you want to clear the database?"):
            self.db = None
            self._reset_app_state()
            self._clear_all_tabs()
            self.info_label.config(text="No database loaded")
            self._update_status("Database cleared")

    def _clear_all_tabs(self):
        """Clear all tab contents."""
        for item in self.sim_tree.get_children():
            self.sim_tree.delete(item)
        self.prop_text.delete(1.0, tk.END)
        
        self.active_filters_text.delete(1.0, tk.END)
        self.results_text.delete(1.0, tk.END)
        self.progress_var.set("Ready")
        
        self.plot_fig.clear()
        self.plot_canvas.draw()
        self.hist_fig.clear()
        self.hist_canvas.draw()
        self.individual_fig.clear()
        self.individual_canvas.draw()

    # ==================== OVERVIEW TAB ====================

    def _create_overview_tab(self):
        """Create the Overview tab."""
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="Overview")
        
        left_frame = ttk.LabelFrame(tab, text="Simulations", padding="5")
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        
        self.sim_tree = ttk.Treeview(left_frame, columns=("Path",), show="headings", height=20)
        self.sim_tree.heading("Path", text="Simulation Path")
        self.sim_tree.column("Path", width=500)
        
        sim_scroll = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.sim_tree.yview)
        self.sim_tree.configure(yscrollcommand=sim_scroll.set)
        
        self.sim_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sim_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        right_frame = ttk.LabelFrame(tab, text="Properties Summary", padding="5")
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        self.prop_text = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD, height=20)
        self.prop_text.pack(fill=tk.BOTH, expand=True)
        
        ttk.Button(tab, text="Refresh Overview", command=self._update_overview_tab).grid(row=1, column=0, columnspan=2, pady=5)
        
        tab.columnconfigure(0, weight=1)
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(0, weight=1)

    def _update_overview_tab(self):
        """Update the overview tab with current database information."""
        if not self.db:
            return
        
        for item in self.sim_tree.get_children():
            self.sim_tree.delete(item)
        self.prop_text.delete(1.0, tk.END)
        
        simulations = self.db.where(**self.search_criteria)
        total_count = len(simulations)
        
        for sim in simulations:
            path = sim.get('path', 'Unknown')
            if not os.path.exists(path):
                path = "(Path not found) " + path
            self.sim_tree.insert("", tk.END, values=(path,))
        
        all_keys = self.db.all_keys()
        property_stats = {}
        
        for key in all_keys:
            if key == 'path':
                continue
            unique_values = self.db.unique(key, **self.search_criteria)
            if 1 < len(unique_values) < total_count:
                property_stats[key] = {'unique_count': len(unique_values), 'values': unique_values}
        
        self.prop_text.insert(tk.END, f"Total Displayed Simulations: {total_count}\n")
        self.prop_text.insert(tk.END, "=" * 50 + "\n\n")
        self.prop_text.insert(tk.END, f"Properties with multiple values ({len(property_stats)}):\n\n")
        
        for prop, stats in sorted(property_stats.items()):
            self.prop_text.insert(tk.END, f"{prop}: {stats['unique_count']} unique values\n")
            formatted_vals = [self._format_value(v) for v in stats['values'][:10]]
            self.prop_text.insert(tk.END, f"  Values: {', '.join(formatted_vals)}")
            if len(stats['values']) > 10:
                self.prop_text.insert(tk.END, " ...")
            self.prop_text.insert(tk.END, "\n\n")
        
        self._update_status(f"Overview updated: {total_count} simulations")

    # ==================== FILTER TAB ====================

    def _create_filter_tab(self):
        """Create the Filter tab with dynamic filtering support."""
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="Filter")
        
        mode_frame = ttk.Frame(tab)
        mode_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        ttk.Label(mode_frame, text="Filter Mode:").pack(side=tk.LEFT, padx=(0, 10))
        self.filter_mode_var = tk.StringVar(value="exact")
        for mode, label in [("exact", "Exact Match"), ("condition", "Condition (>, <, ==)"), ("expression", "Custom Expression")]:
            ttk.Radiobutton(mode_frame, text=label, variable=self.filter_mode_var, value=mode, command=self._toggle_filter_mode).pack(side=tk.LEFT, padx=5)
        
        control_frame = ttk.LabelFrame(tab, text="Filter Criteria", padding="5")
        control_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        tab.columnconfigure(0, weight=1)
        
        self.prop_frame = ttk.Frame(control_frame)
        self.prop_frame.pack(fill=tk.X, pady=5)
        ttk.Label(self.prop_frame, text="Property:").pack(side=tk.LEFT)
        self.filter_property_var = tk.StringVar()
        self.filter_property_combo = ttk.Combobox(self.prop_frame, textvariable=self.filter_property_var, state="readonly", width=30)
        self.filter_property_combo.pack(side=tk.LEFT, padx=5)
        self.filter_property_combo.bind("<<ComboboxSelected>>", self._on_property_select)
        
        self.input_container = ttk.Frame(control_frame)
        self.input_container.pack(fill=tk.X, pady=5)
        
        # Exact match frame
        self.exact_frame = ttk.Frame(self.input_container)
        ttk.Label(self.exact_frame, text="Values:").pack(side=tk.LEFT)
        self.filter_values_listbox = tk.Listbox(self.exact_frame, selectmode=tk.MULTIPLE, height=5, width=50)
        self.filter_values_listbox.pack(side=tk.LEFT, padx=5)
        value_scroll = ttk.Scrollbar(self.exact_frame, orient=tk.VERTICAL, command=self.filter_values_listbox.yview)
        self.filter_values_listbox.configure(yscrollcommand=value_scroll.set)
        value_scroll.pack(side=tk.LEFT, fill=tk.Y)
        
        # Condition frame
        self.condition_frame = ttk.Frame(self.input_container)
        ttk.Label(self.condition_frame, text="Operator:").pack(side=tk.LEFT)
        self.cond_op_var = tk.StringVar(value=">")
        ttk.Combobox(self.condition_frame, textvariable=self.cond_op_var, state="readonly", width=5, 
                     values=[">", "<", ">=", "<=", "==", "!="]).pack(side=tk.LEFT, padx=5)
        ttk.Label(self.condition_frame, text="Value:").pack(side=tk.LEFT, padx=(10, 0))
        self.cond_val_var = tk.StringVar()
        ttk.Entry(self.condition_frame, textvariable=self.cond_val_var, width=20).pack(side=tk.LEFT, padx=5)
        
        # Expression frame
        self.expression_frame = ttk.Frame(self.input_container)
        ttk.Label(self.expression_frame, text="Expression:").pack(side=tk.LEFT)
        self.expr_var = tk.StringVar()
        ttk.Entry(self.expression_frame, textvariable=self.expr_var, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Label(self.expression_frame, text="(e.g., Ne_0 > 1e10 and buildup == True)").pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Add Filter", command=self._add_filter).pack(pady=5)
        
        filters_frame = ttk.LabelFrame(tab, text="Active Filters (JSON Format - Copy/Paste)", padding="5")
        filters_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 10))
        tab.rowconfigure(2, weight=1)
        
        self.active_filters_text = scrolledtext.ScrolledText(filters_frame, height=8, wrap=tk.WORD, font=("TkFixedFont", 9))
        self.active_filters_text.pack(fill=tk.BOTH, expand=True)
        
        # Add context menu for copy/paste
        self.active_filters_text.bind("<Button-3>", self._show_filter_context_menu)
        
        button_frame = ttk.Frame(tab)
        button_frame.grid(row=3, column=0, pady=10)
        ttk.Button(button_frame, text="Apply Filter", command=self._apply_filter).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear All Filters", command=self._clear_filters).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset to Full Database", command=self._reset_filter).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="New Temp. Window", command=self._open_temp_window).pack(side=tk.LEFT, padx=5)
        
        self._toggle_filter_mode()

    def _toggle_filter_mode(self):
        """Toggle visibility of input frames based on selected mode."""
        mode = self.filter_mode_var.get()
        self.exact_frame.pack_forget()
        self.condition_frame.pack_forget()
        self.expression_frame.pack_forget()
        
        if mode == "exact":
            self.prop_frame.pack(fill=tk.X, pady=5, before=self.input_container)
            self.exact_frame.pack(fill=tk.X)
        elif mode == "condition":
            self.prop_frame.pack(fill=tk.X, pady=5, before=self.input_container)
            self.condition_frame.pack(fill=tk.X)
        else:  # expression
            self.prop_frame.pack_forget()
            self.expression_frame.pack(fill=tk.X)

    def _on_property_select(self, event=None):
        """Populate values listbox when a property is selected."""
        if not self.db:
            return
        
        prop = self.filter_property_var.get()
        if not prop:
            return
        
        all_values = set()
        for doc in self.db.db.all():
            if prop in doc:
                val = doc[prop]
                all_values.add(tuple(val) if isinstance(val, list) else val)
        
        sorted_vals = sorted(list(all_values), key=lambda x: str(x))
        self.filter_val_map = {self._format_value(v): v for v in sorted_vals}
        
        self.filter_values_listbox.delete(0, tk.END)
        for val_str in self.filter_val_map.keys():
            self.filter_values_listbox.insert(tk.END, val_str)

    def _add_filter(self):
        """Add a filter criterion based on current mode."""
        mode = self.filter_mode_var.get()
        
        try:
            if mode == "exact":
                self._add_exact_filter()
            elif mode == "condition":
                self._add_condition_filter()
            else:  # expression
                self._add_expression_filter()
            
            self._update_active_filters_display()
        except ValueError as e:
            messagebox.showwarning("Warning", str(e))

    def _add_exact_filter(self):
        """Add exact match filter."""
        prop = self.filter_property_var.get()
        if not prop:
            raise ValueError("Please select a property to filter on")
        
        selected_indices = self.filter_values_listbox.curselection()
        if not selected_indices:
            raise ValueError("Please select at least one value")
        
        selected_strings = [self.filter_values_listbox.get(i) for i in selected_indices]
        exact_values = [self.filter_val_map[val_str] for val_str in selected_strings]
        
        for f in self.active_filters_list:
            if f["type"] == "exact" and f["property"] == prop:
                f["values"] = list(set(f["values"] + exact_values))
                return
        
        self.active_filters_list.append({"type": "exact", "property": prop, "values": exact_values})

    def _add_condition_filter(self):
        """Add condition filter."""
        prop = self.filter_property_var.get()
        if not prop:
            raise ValueError("Please select a property for the condition")
        
        val_str = self.cond_val_var.get().strip()
        if not val_str:
            raise ValueError("Please provide a value for the condition")
        
        try:
            val = float(val_str)
            if val.is_integer():
                val = int(val)
        except ValueError:
            val = val_str
        
        op = self.cond_op_var.get()
        self.active_filters_list.append({"type": "condition", "property": prop, "operator": op, "value": val})

    def _add_expression_filter(self):
        """Add expression filter."""
        expr = self.expr_var.get().strip()
        if not expr:
            raise ValueError("Please enter an expression")
        
        self.active_filters_list.append({"type": "expression", "expr": expr})

    def _update_active_filters_display(self):
        """Update the active filters display in JSON format."""
        self.active_filters_text.delete(1.0, tk.END)
        
        if not self.active_filters_list:
            self.active_filters_text.insert(tk.END, "[]")
            return
        
        json_str = FilterDefinition.serialize_all(self.active_filters_list)
        self.active_filters_text.insert(tk.END, json_str)

    def _show_filter_context_menu(self, event):
        """Show context menu for filter textbox."""
        context_menu = tk.Menu(self.root, tearoff=0)
        context_menu.add_command(label="Copy", command=self._copy_filters)
        context_menu.add_command(label="Paste", command=self._paste_filters)
        context_menu.add_separator()
        context_menu.add_command(label="Clear", command=self._clear_filters)
        
        try:
            context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            context_menu.grab_release()

    def _copy_filters(self):
        """Copy filter definitions to clipboard."""
        json_str = FilterDefinition.serialize_all(self.active_filters_list)
        self.root.clipboard_clear()
        self.root.clipboard_append(json_str)
        self._update_status("Filters copied to clipboard")

    def _paste_filters(self):
        """Paste filter definitions from clipboard."""
        try:
            clipboard_data = self.root.clipboard_get()
            new_filters = FilterDefinition.deserialize_all(clipboard_data)
            self.active_filters_list.extend(new_filters)
            self._update_active_filters_display()
            self._update_status(f"Pasted {len(new_filters)} filter(s)")
        except ValueError as e:
            messagebox.showerror("Error", f"Failed to paste filters: {str(e)}")

    def _build_search_criteria(self):
        """Build query dictionary from active filters."""
        self.search_criteria = {}
        for i, f_data in enumerate(self.active_filters_list):
            if f_data["type"] == "exact":
                self.search_criteria[f_data["property"]] = WhereIn(*f_data["values"])
            elif f_data["type"] == "condition":
                self.search_criteria[f"_cond_{i}"] = self._make_condition(f_data["property"], f_data["operator"], f_data["value"])
            else:  # expression
                self.search_criteria[f"_expr_{i}"] = f_data["expr"]

    @staticmethod
    def _make_condition(prop: str, op: str, val: Any) -> Callable:
        """Create a condition function for filtering."""
        def cond_fn(result):
            if prop not in result:
                return False
            res_val = result[prop]
            try:
                if op == ">":
                    return res_val > val
                elif op == "<":
                    return res_val < val
                elif op == ">=":
                    return res_val >= val
                elif op == "<=":
                    return res_val <= val
                elif op == "==":
                    return res_val == val
                elif op == "!=":
                    return res_val != val
            except TypeError:
                return False
            return False
        return cond_fn

    def _apply_filter(self):
        """Apply the global search criteria."""
        if not self.db:
            messagebox.showwarning("Warning", "No database loaded")
            return
        
        if not self.active_filters_list:
            messagebox.showwarning("Warning", "No filters defined")
            return
        
        try:
            self._build_search_criteria()
            filtered_sims = self.db.where(**self.search_criteria)
            self.current_models = [ECModel(self.db.db, doc) for doc in filtered_sims]
            
            self._update_overview_tab()
            self._update_individual_sim_list()
            
            self._update_status(f"Filter applied: {len(self.current_models)} simulations match.")
            messagebox.showinfo("Success", f"Filter applied successfully. {len(self.current_models)} simulations match.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply filter: {str(e)}")

    def _clear_filters(self):
        """Clear all active filters from UI."""
        self.active_filters_list.clear()
        self._update_active_filters_display()
        self._update_status("Filters cleared")

    def _reset_filter(self):
        """Reset search criteria and reload full database."""
        if not self.db:
            return
        
        self._clear_filters()
        self.search_criteria = {}
        self.current_models = [ECModel(self.db.db, doc) for doc in self.db.db.all()]
        self._update_overview_tab()
        self._update_individual_sim_list()
        self._update_status("Database reset to full dataset.")
        messagebox.showinfo("Reset", "Filters cleared. Showing all simulations.")

    def _open_temp_window(self):
        """Open a new GUI window with currently filtered results."""
        if not self.current_models:
            messagebox.showwarning("Warning", "No simulations to display in a new window.")
            return
        
        new_window = tk.Toplevel(self.root)
        new_app = ECAApp(new_window, db=self.db, is_temp=True)
        new_app.current_models = self.current_models.copy()
        new_app._update_overview_tab()
        new_app._update_individual_sim_list()

    def _populate_filter_options(self):
        """Populate the filter property dropdown."""
        if not self.db:
            return
        
        all_keys = self.db.all_keys()
        total_count = len(self.db.where())
        valid_properties = []
        
        for key in all_keys:
            if key != 'path':
                unique_values = self.db.unique(key)
                if 1 < len(unique_values) < total_count:
                    valid_properties.append(key)
        
        self.filter_property_combo['values'] = valid_properties

    # ==================== FITTING TAB ====================

    def _create_fitting_tab(self):
        """Create the Fitting tab."""
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="Fitting")
        
        model_frame = ttk.LabelFrame(tab, text="Fit Model", padding="5")
        model_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        tab.columnconfigure(0, weight=1)
        
        ttk.Label(model_frame, text="Select Fit Model:").pack(side=tk.LEFT)
        self.fit_model_var = tk.StringVar(value="FurmanNoPhoto")
        ttk.Combobox(model_frame, textvariable=self.fit_model_var, state="readonly", width=30,
                     values=list(self.FIT_MODELS.keys())).pack(side=tk.LEFT, padx=5)

        selector_frame = ttk.LabelFrame(tab, text="Data Selection", padding="5")
        selector_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        
        ttk.Label(selector_frame, text="Selector:").pack(side=tk.LEFT)
        self.fit_selector_var = tk.StringVar(value="BunchAverage")
        ttk.Combobox(selector_frame, textvariable=self.fit_selector_var, state="readonly", width=15,
                     values=["BunchAverage", "BeforeBunch"]).pack(side=tk.LEFT, padx=5)
                     
        self.fit_central_density_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(selector_frame, text="Use Central Density", variable=self.fit_central_density_var).pack(side=tk.LEFT, padx=10)
        
        ttk.Label(selector_frame, text="Train:").pack(side=tk.LEFT)
        self.fit_train_var = tk.StringVar(value="-1")
        ttk.Entry(selector_frame, textvariable=self.fit_train_var, width=5).pack(side=tk.LEFT, padx=2)
        
        selection_frame = ttk.LabelFrame(tab, text="Selection", padding="5")
        selection_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 10))
        tab.rowconfigure(2, weight=0)
        
        self.selection_mode = tk.StringVar(value="filtered")
        ttk.Radiobutton(selection_frame, text="Use Filtered Simulations", 
                       variable=self.selection_mode, value="filtered").pack(anchor=tk.W)
        ttk.Radiobutton(selection_frame, text="Use All Simulations", 
                       variable=self.selection_mode, value="all").pack(anchor=tk.W)
        
        progress_frame = ttk.LabelFrame(tab, text="Progress", padding="5")
        progress_frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        self.progress_var = tk.StringVar(value="Ready")
        ttk.Label(progress_frame, textvariable=self.progress_var).pack(fill=tk.X)
        
        button_frame = ttk.Frame(tab)
        button_frame.grid(row=4, column=0, pady=10)
        ttk.Button(button_frame, text="Apply Fit", command=self._apply_fit).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Refit All", command=self._refit_all).pack(side=tk.LEFT, padx=5)
        
        results_frame = ttk.LabelFrame(tab, text="Fitting Results", padding="5")
        results_frame.grid(row=5, column=0, sticky="nsew")
        tab.rowconfigure(5, weight=1)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=15, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True)

    def _get_models_to_fit(self) -> List[ECModel]:
        """Get models to fit based on selection mode."""
        if not self.db:
            return []
        
        if self.selection_mode.get() == "filtered" and self.search_criteria:
            return [ECModel(self.db.db, doc) for doc in self.db.where(**self.search_criteria)]
        else:
            return [ECModel(self.db.db, doc) for doc in self.db.where()]

    def _apply_fit(self, refit: bool = False):
        """Apply the selected fit model to simulations."""
        if not self.db:
            messagebox.showwarning("Warning", "No database loaded")
            return
        
        model_name = self.fit_model_var.get()
        models = self._get_models_to_fit()
        
        if not models:
            messagebox.showwarning("Warning", "No simulations to fit")
            return
            
        selector_type = self.fit_selector_var.get()
        use_cd = self.fit_central_density_var.get()
        try:
            use_train = int(self.fit_train_var.get())
        except ValueError:
            messagebox.showwarning("Warning", "Train must be an integer.")
            return

        SelectorClass = BeforeBunchSelector if selector_type == "BeforeBunch" else BunchAverageSelector
        selector = SelectorClass(use_central_density=use_cd, use_train=use_train)
        
        fit_class = self.FIT_MODELS.get(model_name)
        if not fit_class:
            raise ValueError(f"Unknown fit model: {model_name}")
        
        fit = fit_class(self.db, selector=selector) if model_name == "FurmanNPMC" else fit_class(selector=selector)
        
        self.progress_var.set("Starting fit...")
        self.root.update()
        
        success_count = sum(1 for i, model in enumerate(models)
                            if self._execute_fit(fit, model, refit, i, len(models)))
        
        self._display_fit_results(success_count, len(models))

    def _execute_fit(self, fit, model: ECModel, refit: bool, current: int, total: int) -> bool:
        """Execute a single fit and update progress."""
        try:
            result = fit.fit(model, refit=refit)
        except:
            result = None
        self.progress_var.set(f"Fitting: {current+1}/{total}")
        self.root.update()
        return result is not None

    def _display_fit_results(self, success_count: int, total: int):
        """Display fitting results."""
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Fitting Complete\n{'=' * 50}\n")
        self.results_text.insert(tk.END, f"Total: {total}\nSuccessful: {success_count}\nFailed: {total - success_count}\n")
        self._update_status(f"Fitting complete: {success_count}/{total} successful")

    def _refit_all(self):
        """Refit all simulations with the selected model."""
        if not self.db:
            messagebox.showwarning("Warning", "No database loaded")
            return
        
        if messagebox.askyesno("Confirm", "This will refit all selected simulations. Continue?"):
            self._apply_fit(refit=True)

    # ==================== PLOTTING TAB ====================

    def _create_plotting_tab(self):
        """Create the Plotting tab for database-wide plots."""
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="Database Plots")
        
        control_frame = ttk.LabelFrame(tab, text="Plot Controls", padding="5")
        control_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        tab.columnconfigure(0, weight=1)
        
        axes_frame = ttk.Frame(control_frame)
        axes_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(axes_frame, text="X-axis:").pack(side=tk.LEFT)
        self.plot_x_var = tk.StringVar()
        self.plot_x_combo = ttk.Combobox(axes_frame, textvariable=self.plot_x_var, state="readonly", width=20)
        self.plot_x_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(axes_frame, text="Y-axis:").pack(side=tk.LEFT)
        self.plot_y_var = tk.StringVar()
        self.plot_y_combo = ttk.Combobox(axes_frame, textvariable=self.plot_y_var, state="readonly", width=20)
        self.plot_y_combo.pack(side=tk.LEFT, padx=5)
        
        color_frame = ttk.Frame(control_frame)
        color_frame.pack(fill=tk.X, pady=5)
        ttk.Label(color_frame, text="Color by:").pack(side=tk.LEFT)
        self.plot_color_var = tk.StringVar(value="None")
        self.plot_color_combo = ttk.Combobox(color_frame, textvariable=self.plot_color_var, state="readonly", width=20)
        self.plot_color_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Generate Plot", command=self._generate_plot).pack(pady=10)
        
        plot_frame = ttk.LabelFrame(tab, text="Plot", padding="5")
        plot_frame.grid(row=1, column=0, sticky="nsew")
        tab.rowconfigure(1, weight=1)
        
        self.plot_fig = Figure(figsize=(8, 6), dpi=100)
        self.plot_canvas = FigureCanvasTkAgg(self.plot_fig, master=plot_frame)
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.plot_canvas, plot_frame).update()

    def _populate_plot_options(self):
        """Populate plot axis dropdowns."""
        if not self.db:
            return
        
        plotable_props = self._get_plotable_properties()
        self.plot_x_combo['values'] = plotable_props
        self.plot_y_combo['values'] = plotable_props
        self.plot_color_combo['values'] = ["None"] + plotable_props
        
        if hasattr(self, 'hist_prop_combo'):
            self.hist_prop_combo['values'] = plotable_props

    def _get_plotable_properties(self) -> List[str]:
        """Get list of plotable properties."""
        if not self.db:
            return []
        
        plotable_props = []
        for key in self.db.all_keys():
            if key in ['path', 'processed']:
                continue
            
            try:
                unique_values = self.db.unique(key)
                if len(unique_values) <= 1:
                    continue
                
                if any(isinstance(v, (int, float, np.number)) and not (np.isnan(v) or np.isinf(v)) 
                       for v in unique_values):
                    plotable_props.append(key)
            except Exception:
                continue
        
        return plotable_props

    def _generate_plot(self):
        """Generate a plot using ecaplots.versus_plot."""
        if not self.db:
            messagebox.showwarning("Warning", "No database loaded")
            return
        
        x_prop, y_prop = self.plot_x_var.get(), self.plot_y_var.get()
        color_prop = self.plot_color_var.get() if self.plot_color_var.get() != "None" else None
        
        if not (x_prop and y_prop):
            messagebox.showwarning("Warning", "Please select both X and Y axes")
            return
        
        try:
            self.plot_fig.clear()
            ax = self.plot_fig.add_subplot(111)
            
            db_to_use = self.db.extract(self.db.all_keys(), **self.search_criteria) if self.search_criteria else self.db
            versus_plot(db_to_use, x_prop, y_prop, colorBy=color_prop, size=ax)
            
            self.plot_canvas.draw()
            title_suffix = f" (colored by {color_prop})" if color_prop else ""
            self._update_status(f"Plot generated: {y_prop} vs {x_prop}{title_suffix}")
        except Exception as e:
            messagebox.showerror("Error", f"Plot generation failed: {str(e)}")

    # ==================== HISTOGRAM TAB ====================

    def _create_histogram_tab(self):
        """Create the Histogram tab."""
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="Histogram Plots")
        
        control_frame = ttk.LabelFrame(tab, text="Plot Controls", padding="5")
        control_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        tab.columnconfigure(0, weight=1)
        
        settings_frame = ttk.Frame(control_frame)
        settings_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(settings_frame, text="Property:").pack(side=tk.LEFT)
        self.hist_prop_var = tk.StringVar()
        self.hist_prop_combo = ttk.Combobox(settings_frame, textvariable=self.hist_prop_var, state="readonly", width=20)
        self.hist_prop_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(settings_frame, text="Bins:").pack(side=tk.LEFT, padx=(10, 0))
        self.hist_bins_var = tk.StringVar(value="auto")
        ttk.Entry(settings_frame, textvariable=self.hist_bins_var, width=10).pack(side=tk.LEFT, padx=5)
        
        self.hist_log_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(settings_frame, text="Log Y-axis", variable=self.hist_log_var).pack(side=tk.LEFT, padx=10)
        
        ttk.Button(control_frame, text="Generate Histogram", command=self._generate_histogram).pack(pady=10)
        
        plot_frame = ttk.LabelFrame(tab, text="Plot", padding="5")
        plot_frame.grid(row=1, column=0, sticky="nsew")
        tab.rowconfigure(1, weight=1)
        
        self.hist_fig = Figure(figsize=(8, 6), dpi=100)
        self.hist_canvas = FigureCanvasTkAgg(self.hist_fig, master=plot_frame)
        self.hist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.hist_canvas, plot_frame).update()

    def _generate_histogram(self):
        """Generate a histogram using ecaplots.histogram_plot."""
        if not self.db:
            messagebox.showwarning("Warning", "No database loaded")
            return
        
        prop = self.hist_prop_var.get()
        if not prop:
            messagebox.showwarning("Warning", "Please select a property")
            return
        
        bins_val = self.hist_bins_var.get().strip()
        if bins_val.lower() != 'auto':
            try:
                bins_val = int(bins_val)
                if bins_val <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showwarning("Warning", "Bins must be 'auto' or a positive integer.")
                return
        
        try:
            self.hist_fig.clear()
            ax = self.hist_fig.add_subplot(111)
            
            db_to_use = self.db.extract(self.db.all_keys(), **self.search_criteria) if self.search_criteria else self.db
            histogram_plot(db_to_use, prop, bins=bins_val, log_y=self.hist_log_var.get(), size=ax)
            
            self.hist_canvas.draw()
            self._update_status(f"Histogram generated: {prop}")
        except Exception as e:
            messagebox.showerror("Error", f"Histogram generation failed: {str(e)}")

    # ==================== INDIVIDUAL PLOT TAB ====================

    def _create_individual_plot_tab(self):
        """Create the Individual Plot tab."""
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="Individual Plot")
        
        left_frame = ttk.LabelFrame(tab, text="Select Simulation", padding="5")
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)
        
        col_frame = ttk.Frame(left_frame)
        col_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(col_frame, text="Add Column:").pack(side=tk.LEFT)
        self.indiv_col_var = tk.StringVar()
        self.indiv_col_combo = ttk.Combobox(col_frame, textvariable=self.indiv_col_var, state="readonly", width=15)
        self.indiv_col_combo.pack(side=tk.LEFT, padx=5)
        ttk.Button(col_frame, text="Add", command=self._add_indiv_column).pack(side=tk.LEFT, padx=2)
        ttk.Button(col_frame, text="Reset", command=self._reset_indiv_columns).pack(side=tk.LEFT, padx=2)
        
        self.individual_sim_tree = ttk.Treeview(left_frame, columns=self.individual_columns, show="headings", height=15)
        self.individual_sim_tree.heading("doc_id", text="Doc ID")
        self.individual_sim_tree.column("doc_id", width=80)
        sim_scroll = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.individual_sim_tree.yview)
        self.individual_sim_tree.configure(yscrollcommand=sim_scroll.set)
        self.individual_sim_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sim_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.individual_sim_tree.bind("<<TreeviewSelect>>", lambda e: self._plot_individual_simulation())
        
        right_frame = ttk.Frame(tab)
        right_frame.grid(row=0, column=1, sticky="nsew")
        tab.columnconfigure(1, weight=1)
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)
        
        fit_frame = ttk.LabelFrame(right_frame, text="Plot Configuration", padding="5")
        fit_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        ttk.Label(fit_frame, text="Apply Fits:").pack(side=tk.LEFT)
        
        self.individual_fit_vars = {}
        for fit_name in self.FIT_MODELS.keys():
            var = tk.BooleanVar(value=False)
            self.individual_fit_vars[fit_name] = var
            # Bind the checkbutton to immediately redraw the plot
            ttk.Checkbutton(fit_frame, text=fit_name, variable=var, command=self._plot_individual_simulation).pack(side=tk.LEFT, padx=5)
        
        self.individual_central_density_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(fit_frame, text="Use Central Density", variable=self.individual_central_density_var, command=self._plot_individual_simulation).pack(side=tk.LEFT, padx=10)

        # Replace text entry with Combobox for selecting train
        ttk.Label(fit_frame, text="View Train:").pack(side=tk.LEFT, padx=(5, 2))
        self.individual_train_var = tk.StringVar(value="All")
        self.individual_train_combo = ttk.Combobox(fit_frame, textvariable=self.individual_train_var, state="readonly", width=8)
        self.individual_train_combo.pack(side=tk.LEFT, padx=2)
        self.individual_train_combo.bind("<<ComboboxSelected>>", lambda e: self._plot_individual_simulation())
        
        ttk.Label(fit_frame, text="Max X:").pack(side=tk.LEFT, padx=(5, 2))
        self.individual_max_x_var = tk.StringVar()
        ttk.Entry(fit_frame, textvariable=self.individual_max_x_var, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(fit_frame, text="Replot", command=self._plot_individual_simulation).pack(side=tk.LEFT, padx=10)
        
        plot_frame = ttk.LabelFrame(right_frame, text="Simulation Plot", padding="5")
        plot_frame.grid(row=1, column=0, sticky="nsew")
        
        self.individual_fig = Figure(figsize=(10, 6), dpi=100)
        self.individual_canvas = FigureCanvasTkAgg(self.individual_fig, master=plot_frame)
        self.individual_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.individual_canvas, plot_frame).update()

    def _populate_individual_options(self):
        """Populate the column selection combobox."""
        if self.db:
            self.indiv_col_combo['values'] = [k for k in self.db.all_keys() if k != "path"]

    def _add_indiv_column(self):
        """Add a column to the individual simulation list."""
        new_col = self.indiv_col_var.get()
        if new_col and new_col not in self.individual_columns:
            self.individual_columns.append(new_col)
            self._update_individual_sim_list()

    def _reset_indiv_columns(self):
        """Reset columns to default."""
        self.individual_columns = ["doc_id"]
        self._update_individual_sim_list()

    def _update_individual_sim_list(self):
        """Update the individual simulation list."""
        for item in self.individual_sim_tree.get_children():
            self.individual_sim_tree.delete(item)
        
        if not self.db:
            return
        
        self.individual_sim_tree.configure(columns=self.individual_columns)
        for col in self.individual_columns:
            self.individual_sim_tree.heading(col, text=col)
            self.individual_sim_tree.column(col, width=120 if col != "doc_id" else 60)
        
        for sim in self.db.where(**self.search_criteria):
            values = [sim.doc_id if col == "doc_id" else self._format_value(sim.get(col, "N/A")) 
                     for col in self.individual_columns]
            self.individual_sim_tree.insert("", tk.END, values=values)

    def _plot_individual_simulation(self, *args):
        """Plot selected simulation(s)."""
        selections = self.individual_sim_tree.selection()
        if not (selections and self.db):
            return
        
        self.individual_fig.clear()
        ax = self.individual_fig.add_subplot(111)
        
        # Collect selected fit models
        selected_fits = [name for name, var in self.individual_fit_vars.items() if var.get()]
        plotted_count = 0
        
        # Determine train view logic and populate dropdown if exactly 1 simulation is selected
        if len(selections) == 1:
            try:
                doc_id_idx = self.individual_columns.index("doc_id")
                doc_id = int(self.individual_sim_tree.item(selections[0])['values'][doc_id_idx])
                model = ECModel(self.db.db, doc_id)
                num_trains = len(model.train_times)
                train_options = ["All"] + [str(i) for i in range(num_trains)]
                
                # Check if we should update default selection (due to model or fit toggles)
                if getattr(self, '_last_doc_id', None) != doc_id or getattr(self, '_last_fits', None) != selected_fits:
                    self.individual_train_combo['values'] = train_options
                    if selected_fits:
                        fit_class = self.FIT_MODELS[selected_fits[0]]
                        fit_inst = fit_class(self.db) if selected_fits[0] == "FurmanNPMC" else fit_class()
                        fit_train = fit_inst._mget("train", model, -1)
                        if fit_train < 0:
                            fit_train += num_trains
                        if 0 <= fit_train < num_trains:
                            self.individual_train_var.set(str(fit_train))
                    else:
                        self.individual_train_var.set("All")
                        
                    self._last_doc_id = doc_id
                    self._last_fits = selected_fits.copy()
            except (ValueError, IndexError, AttributeError):
                pass

        # Parse selected train
        train_str = self.individual_train_var.get().strip()
        plot_train = "All" if not train_str or train_str.lower() == "all" else int(train_str)
        
        for selection in selections:
            try:
                doc_id_idx = self.individual_columns.index("doc_id")
                doc_id = int(self.individual_sim_tree.item(selection)['values'][doc_id_idx])
            except (ValueError, IndexError):
                continue
            
            model = ECModel(self.db.db, doc_id)
            path_exists = os.path.exists(model.path) and os.path.exists(os.path.join(model.path, "Pyecltest.mat"))
            
            fits_to_plot = []
            for fit_model_name in selected_fits:
                fit_class = self.FIT_MODELS.get(fit_model_name)
                if fit_class:
                    fit_inst = fit_class(self.db) if fit_model_name == "FurmanNPMC" else fit_class()
                    fits_to_plot.append(fit_inst)
            
            if not (path_exists or fits_to_plot):
                continue
            
            cd_param = self.individual_central_density_var.get() if path_exists else None
            
            max_x_str = self.individual_max_x_var.get().strip()
            try:
                fit_max_x = float(max_x_str) if max_x_str else (model.cutoff / model.bunch_step) * 1.25
                if len(selections) == 1:
                    self.individual_max_x_var.set(f"{fit_max_x:.1f}")
            except (ValueError, AttributeError):
                fit_max_x = 300.0
            
            model_plot(model=model, fits=fits_to_plot, size=ax, show_error=True, 
                       central_density=cd_param, fit_maxX=fit_max_x, label=str(doc_id),
                       train=plot_train)
            plotted_count += 1
        
        if plotted_count > 0:
            self.individual_fig.tight_layout()
            self.individual_canvas.draw()
            self._update_status(f"Plotted {plotted_count} simulation(s)")


def main():
    """Main entry point for the application."""
    root = tk.Tk()
    app = ECAApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()