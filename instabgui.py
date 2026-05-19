"""
INSTAB GUI - Instability Analysis Graphical Interface
A GUI application for browsing, filtering, and visualizing PyECLOUD/PyHEADTAIL instability simulation data.
Mirrors the architecture and functionality of ecagui.py for instability-specific analysis.
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
    from eca import SimDB, InstabilityModel, InstabilityDBFolder, WhereIn
    from ecaplots import (
        instability_grid_plot,
        growth_rate_vs_density_plot,
        blowup_time_vs_strength_plot,
        intrabunch_mode_heatmap,
        instability_mode_evolution_plot
    )
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
except ImportError as e:
    print(f"Error importing Instability modules: {e}")
    print("Make sure eca.py and ecaplots.py are in the same directory or PYTHONPATH")
    sys.exit(1)


class FilterDefinition:
    """Serializable filter definition for copy-paste support (reused from ecagui)."""

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


class InstabApp:
    """Main application class for Instability Analysis GUI."""

    # Common instability-related keys for quick filtering
    INSTAB_KEYS = [
        "growth_rate_centroid", "growth_rate_mode", "dominant_mode_idx",
        "tune_centroid", "blowup_turn_first", "max_emittance_ratio",
        "instability_threshold", "n_turns", "n_slices", "element",
        "init_unif_edens_dip", "strength_factor"
    ]

    def __init__(self, root: tk.Tk, db: Optional[SimDB] = None, is_temp: bool = False):
        self.root = root
        self.is_temp = is_temp
        self.db: Optional[SimDB] = db
        self.search_criteria: Dict[str, Any] = {}
        self.current_models: List[InstabilityModel] = []
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
        self.root.title(f"{title_prefix}Instab GUI - Instability Analysis")
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
            if abs_val >= 1000 or (abs_val < 1e-4 and abs_val > 0):
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
        self._create_analysis_tab()
        self._create_grid_tab()
        self._create_growth_rate_tab()
        self._create_blowup_tab()
        self._create_mode_evolution_tab()
        self._create_heatmap_tab()
        self._create_versus_plot_tab()
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
            title="Select Instability Database File",
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
        self.individual_sim_tree.delete(*self.individual_sim_tree.get_children())
        self.progress_var.set("Ready")
        
        # Clear all plot canvases
        for fig_attr in ['grid_fig', 'growth_fig', 'blowup_fig', 'mode_fig', 'heatmap_fig', 
                        'versus_fig', 'hist_fig', 'individual_fig']:
            if hasattr(self, fig_attr):
                getattr(self, fig_attr).clear()
                canvas_attr = fig_attr.replace('_fig', '_canvas')
                if hasattr(self, canvas_attr):
                    getattr(self, canvas_attr).draw()

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
        ttk.Label(self.expression_frame, text="(e.g., growth_rate_centroid > 0.01 and n_turns > 5000)").pack(side=tk.LEFT, padx=5)
        
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
            self.current_models = [InstabilityModel(self.db.db, doc) for doc in filtered_sims]
            
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
        self.current_models = [InstabilityModel(self.db.db, doc) for doc in self.db.db.all()]
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
        new_app = InstabApp(new_window, db=self.db, is_temp=True)
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

    # ==================== ANALYSIS TAB ====================

    def _create_analysis_tab(self):
        """Create the Analysis tab for running InstabilityModel.analyze() on filtered data."""
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="Analysis")
        
        selection_frame = ttk.LabelFrame(tab, text="Selection", padding="5")
        selection_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        tab.columnconfigure(0, weight=1)
        
        self.analysis_mode = tk.StringVar(value="filtered")
        ttk.Radiobutton(selection_frame, text="Analyze Filtered Simulations", 
                       variable=self.analysis_mode, value="filtered").pack(anchor=tk.W)
        ttk.Radiobutton(selection_frame, text="Analyze All Simulations", 
                       variable=self.analysis_mode, value="all").pack(anchor=tk.W)
        
        progress_frame = ttk.LabelFrame(tab, text="Progress", padding="5")
        progress_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self.progress_var = tk.StringVar(value="Ready")
        ttk.Label(progress_frame, textvariable=self.progress_var).pack(fill=tk.X)
        
        button_frame = ttk.Frame(tab)
        button_frame.grid(row=2, column=0, pady=10)
        ttk.Button(button_frame, text="Analyze Selected", command=self._run_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Analyze All", command=self._run_analysis_all).pack(side=tk.LEFT, padx=5)
        
        results_frame = ttk.LabelFrame(tab, text="Analysis Results", padding="5")
        results_frame.grid(row=3, column=0, sticky="nsew")
        tab.rowconfigure(3, weight=1)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=15, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True)

    def _get_models_to_analyze(self) -> List[InstabilityModel]:
        """Get models to analyze based on selection mode."""
        if not self.db:
            return []
        
        if self.analysis_mode.get() == "filtered" and self.search_criteria:
            return self.db.where(**self.search_criteria)
        else:
            return self.db.db.all()

    def _run_analysis(self):
        """Run analysis on selected simulations."""
        if not self.db:
            messagebox.showwarning("Warning", "No database loaded")
            return
        
        models = self._get_models_to_analyze()
        
        if not models:
            messagebox.showwarning("Warning", "No simulations to analyze")
            return
        
        if messagebox.askyesno("Confirm", f"Run analysis on {len(models)} simulation(s)? This may take a while."):
            self._execute_analysis(models)

    def _run_analysis_all(self):
        """Run analysis on all simulations in the database."""
        if not self.db:
            messagebox.showwarning("Warning", "No database loaded")
            return
        
        models = self.db.db.all()
        
        if not models:
            messagebox.showwarning("Warning", "No simulations in database")
            return
        
        if messagebox.askyesno("Confirm", f"Run analysis on ALL {len(models)} simulations? This may take a long time."):
            self._execute_analysis(models)

    def _execute_analysis(self, models: List):
        """Execute analysis on a list of models."""
        self.results_text.delete(1.0, tk.END)
        self.progress_var.set("Starting analysis...")
        self.root.update()
        
        success_count = 0
        failed_count = 0
        
        for i, model in enumerate(models):
            try:
                model = InstabilityModel(self.db, model)
                self.progress_var.set(f"Analyzing: {i+1}/{len(models)} - Doc ID: {model.doc_id}")
                self.root.update()
                
                # Run the analyze() method which generates all properties
                model.run_analysis()
                success_count += 1
                
            except Exception as e:
                failed_count += 1
                self.results_text.insert(tk.END, f"Error analyzing Doc ID {model.doc_id}: {str(e)}\n")
        
        # Reload the database to pick up new properties
        if success_count > 0:
            try:
                self._update_status("Reloading database with new properties...")
                self.root.update()
                # Refresh all tabs to show new data
                self._refresh_all_tabs()
            except Exception as e:
                self.results_text.insert(tk.END, f"\nError reloading database: {str(e)}\n")
        
        self._display_analysis_results(success_count, len(models))

    def _display_analysis_results(self, success_count: int, total: int):
        """Display analysis results."""
        result_summary = f"Analysis Complete\n{'=' * 50}\n"
        result_summary += f"Total: {total}\nSuccessful: {success_count}\nFailed: {total - success_count}\n"
        
        self.results_text.insert(1.0, result_summary)
        self.progress_var.set(f"Analysis complete: {success_count}/{total} successful")
        self._update_status(f"Analysis complete: {success_count}/{total} successful")

    # ==================== GRID TAB ====================

    def _create_grid_tab(self):
        """Create the Grid visualization tab."""
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="Instability Grid")

        control_frame = ttk.LabelFrame(tab, text="Grid Configuration", padding="5")
        control_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        tab.columnconfigure(0, weight=1)

        ttk.Label(control_frame, text="Max Simulations:").pack(side=tk.LEFT, padx=(0, 10))
        self.grid_max_var = tk.StringVar(value="5")
        ttk.Entry(control_frame, textvariable=self.grid_max_var, width=5).pack(side=tk.LEFT, padx=5)

        ttk.Label(control_frame, text="Colormap:").pack(side=tk.LEFT, padx=(10, 0))
        self.grid_cmap_var = tk.StringVar(value="viridis")
        cmap_options = ["viridis", "plasma", "inferno", "magma", "cividis", "coolwarm"]
        ttk.Combobox(control_frame, textvariable=self.grid_cmap_var, state="readonly", values=cmap_options, width=15).pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text="Generate Grid Plot", command=self._generate_grid_plot).pack(side=tk.LEFT, padx=20)

        plot_frame = ttk.LabelFrame(tab, text="Evolution Grid", padding="5")
        plot_frame.grid(row=1, column=0, sticky="nsew")
        tab.rowconfigure(1, weight=1)

        self.grid_fig = Figure(figsize=(14, 16), dpi=100)
        self.grid_canvas = FigureCanvasTkAgg(self.grid_fig, master=plot_frame)
        self.grid_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.grid_canvas, plot_frame).update()

    def _generate_grid_plot(self):
        """Generate and display the grid plot."""
        if not self.db:
            messagebox.showwarning("Warning", "No database loaded")
            return

        try:
            max_plots = int(self.grid_max_var.get())
        except ValueError:
            messagebox.showerror("Error", "Max Simulations must be an integer.")
            return

        cmap_name = self.grid_cmap_var.get()
        try:
            colormap = plt.get_cmap(cmap_name)
        except:
            colormap = plt.cm.viridis

        self.grid_fig.clear()

        try:
            self._update_status("Generating grid plot...")
            self.root.update()
            
            db_to_use = self.db.extract(self.db.all_keys(), **self.search_criteria) if self.search_criteria else self.db
            instability_grid_plot(
                db=db_to_use,
                size=(14, 16),
                max_plots=max_plots,
                colormap=colormap
            )
            self.grid_canvas.draw()
            self._update_status("Grid plot generated")
        except Exception as e:
            messagebox.showerror("Error", f"Grid plot generation failed: {str(e)}")
            self._update_status("Error generating grid plot")

    # ==================== GROWTH RATE TAB ====================

    def _create_growth_rate_tab(self):
        """Create the Growth Rate vs Density tab."""
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="Growth Rate vs Density")

        control_frame = ttk.LabelFrame(tab, text="Plot Configuration", padding="5")
        control_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        tab.columnconfigure(0, weight=1)

        ttk.Label(control_frame, text="Density Key:").pack(side=tk.LEFT, padx=(0, 10))
        self.growth_density_var = tk.StringVar(value="init_unif_edens_dip")
        self.growth_density_combo = ttk.Combobox(control_frame, textvariable=self.growth_density_var, state="readonly", width=20)
        self.growth_density_combo.pack(side=tk.LEFT, padx=5)

        ttk.Label(control_frame, text="Growth Key:").pack(side=tk.LEFT, padx=(10, 0))
        self.growth_rate_var = tk.StringVar(value="growth_rate_mode")
        self.growth_rate_combo = ttk.Combobox(control_frame, textvariable=self.growth_rate_var, state="readonly", width=20)
        self.growth_rate_combo.pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text="Generate Plot", command=self._generate_growth_plot).pack(side=tk.LEFT, padx=20)

        plot_frame = ttk.LabelFrame(tab, text="Growth Rate Plot", padding="5")
        plot_frame.grid(row=1, column=0, sticky="nsew")
        tab.rowconfigure(1, weight=1)

        self.growth_fig = Figure(figsize=(8, 6), dpi=100)
        self.growth_canvas = FigureCanvasTkAgg(self.growth_fig, master=plot_frame)
        self.growth_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.growth_canvas, plot_frame).update()

    def _generate_growth_plot(self):
        """Generate and display the growth rate plot."""
        if not self.db:
            messagebox.showwarning("Warning", "No database loaded")
            return

        density_key = self.growth_density_var.get()
        growth_key = self.growth_rate_var.get()

        if not (density_key and growth_key):
            messagebox.showwarning("Warning", "Please select both density and growth keys")
            return

        self.growth_fig.clear()
        try:
            self._update_status("Generating growth rate plot...")
            self.root.update()
            
            db_to_use = self.db.extract(self.db.all_keys(), **self.search_criteria) if self.search_criteria else self.db
            growth_rate_vs_density_plot(
                db=db_to_use,
                density_key=density_key,
                growth_key=growth_key,
                size=(8, 6))
            self.growth_canvas.draw()
            self._update_status("Growth rate plot generated")
        except Exception as e:
            messagebox.showerror("Error", f"Growth rate plot generation failed: {str(e)}")
            self._update_status("Error generating growth rate plot")

    # ==================== BLOWUP TAB ====================

    def _create_blowup_tab(self):
        """Create the Blow-up Time vs Strength tab."""
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="Blow-up Time vs Strength")

        control_frame = ttk.LabelFrame(tab, text="Plot Configuration", padding="5")
        control_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        tab.columnconfigure(0, weight=1)

        ttk.Label(control_frame, text="Strength Key:").pack(side=tk.LEFT, padx=(0, 10))
        self.blowup_strength_var = tk.StringVar(value="strength_factor")
        self.blowup_strength_combo = ttk.Combobox(control_frame, textvariable=self.blowup_strength_var, state="readonly", width=20)
        self.blowup_strength_combo.pack(side=tk.LEFT, padx=5)

        ttk.Label(control_frame, text="Turns Key:").pack(side=tk.LEFT, padx=(10, 0))
        self.blowup_turns_var = tk.StringVar(value="blowup_turn_first")
        self.blowup_turns_combo = ttk.Combobox(control_frame, textvariable=self.blowup_turns_var, state="readonly", width=20)
        self.blowup_turns_combo.pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text="Generate Plot", command=self._generate_blowup_plot).pack(side=tk.LEFT, padx=20)

        plot_frame = ttk.LabelFrame(tab, text="Blow-up Plot", padding="5")
        plot_frame.grid(row=1, column=0, sticky="nsew")
        tab.rowconfigure(1, weight=1)

        self.blowup_fig = Figure(figsize=(8, 6), dpi=100)
        self.blowup_canvas = FigureCanvasTkAgg(self.blowup_fig, master=plot_frame)
        self.blowup_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.blowup_canvas, plot_frame).update()

    def _generate_blowup_plot(self):
        """Generate and display the blow-up plot."""
        if not self.db:
            messagebox.showwarning("Warning", "No database loaded")
            return

        strength_key = self.blowup_strength_var.get()
        turns_key = self.blowup_turns_var.get()

        if not (strength_key and turns_key):
            messagebox.showwarning("Warning", "Please select both strength and turns keys")
            return

        self.blowup_fig.clear()
        try:
            self._update_status("Generating blow-up plot...")
            self.root.update()
            
            db_to_use = self.db.extract(self.db.all_keys(), **self.search_criteria) if self.search_criteria else self.db
            blowup_time_vs_strength_plot(
                db=db_to_use,
                strength_key=strength_key,
                turns_key=turns_key,
                size=(8, 6))
            self.blowup_canvas.draw()
            self._update_status("Blow-up plot generated")
        except Exception as e:
            messagebox.showerror("Error", f"Blow-up plot generation failed: {str(e)}")
            self._update_status("Error generating blow-up plot")

    # ==================== MODE EVOLUTION TAB ====================

    def _create_mode_evolution_tab(self):
        """Create the Mode Evolution tab."""
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="Mode Evolution")

        control_frame = ttk.LabelFrame(tab, text="Simulation Selection", padding="5")
        control_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        tab.columnconfigure(0, weight=1)

        ttk.Label(control_frame, text="Model Index:").pack(side=tk.LEFT, padx=(0, 10))
        self.mode_idx_var = tk.StringVar(value="0")
        ttk.Entry(control_frame, textvariable=self.mode_idx_var, width=5).pack(side=tk.LEFT, padx=5)

        ttk.Label(control_frame, text="Colormap:").pack(side=tk.LEFT, padx=(10, 0))
        self.mode_cmap_var = tk.StringVar(value="viridis")
        cmap_options = ["viridis", "plasma", "inferno", "magma", "cividis", "RdBu_r"]
        ttk.Combobox(control_frame, textvariable=self.mode_cmap_var, state="readonly", values=cmap_options, width=15).pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text="Generate Spectrogram", command=self._generate_mode_plot).pack(side=tk.LEFT, padx=20)

        plot_frame = ttk.LabelFrame(tab, text="Mode Evolution Spectrogram", padding="5")
        plot_frame.grid(row=1, column=0, sticky="nsew")
        tab.rowconfigure(1, weight=1)

        self.mode_fig = Figure(figsize=(12, 8), dpi=100)
        self.mode_canvas = FigureCanvasTkAgg(self.mode_fig, master=plot_frame)
        self.mode_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.mode_canvas, plot_frame).update()

    def _generate_mode_plot(self):
        """Generate and display the mode evolution plot."""
        if not self.db:
            messagebox.showwarning("Warning", "No database loaded")
            return

        try:
            model_idx = int(self.mode_idx_var.get())
        except ValueError:
            messagebox.showerror("Error", "Model Index must be an integer.")
            return

        cmap_name = self.mode_cmap_var.get()
        try:
            colormap = plt.get_cmap(cmap_name)
        except:
            colormap = plt.cm.viridis

        self.mode_fig.clear()
        try:
            self._update_status("Generating mode evolution plot...")
            self.root.update()
            
            db_to_use = self.db.extract(self.db.all_keys(), **self.search_criteria) if self.search_criteria else self.db
            instability_mode_evolution_plot(
                db=db_to_use,
                model_idx=model_idx,
                size=(12, 8),
                colormap=colormap)
            self.mode_canvas.draw()
            self._update_status("Mode evolution plot generated")
        except Exception as e:
            messagebox.showerror("Error", f"Mode evolution plot generation failed: {str(e)}")
            self._update_status("Error generating mode evolution plot")

    # ==================== HEATMAP TAB ====================

    def _create_heatmap_tab(self):
        """Create the Intra-bunch Heatmap tab."""
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="Intra-bunch Heatmap")

        control_frame = ttk.LabelFrame(tab, text="Simulation Selection", padding="5")
        control_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        tab.columnconfigure(0, weight=1)

        ttk.Label(control_frame, text="Model Index:").pack(side=tk.LEFT, padx=(0, 10))
        self.heatmap_idx_var = tk.StringVar(value="0")
        ttk.Entry(control_frame, textvariable=self.heatmap_idx_var, width=5).pack(side=tk.LEFT, padx=5)

        ttk.Label(control_frame, text="Colormap:").pack(side=tk.LEFT, padx=(10, 0))
        self.heatmap_cmap_var = tk.StringVar(value="RdBu_r")
        cmap_options = ["RdBu_r", "viridis", "plasma", "coolwarm", "seismic"]
        ttk.Combobox(control_frame, textvariable=self.heatmap_cmap_var, state="readonly", values=cmap_options, width=15).pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text="Generate Heatmap", command=self._generate_heatmap_plot).pack(side=tk.LEFT, padx=20)

        plot_frame = ttk.LabelFrame(tab, text="Intra-bunch Mode Structure", padding="5")
        plot_frame.grid(row=1, column=0, sticky="nsew")
        tab.rowconfigure(1, weight=1)

        self.heatmap_fig = Figure(figsize=(10, 6), dpi=100)
        self.heatmap_canvas = FigureCanvasTkAgg(self.heatmap_fig, master=plot_frame)
        self.heatmap_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.heatmap_canvas, plot_frame).update()

    def _generate_heatmap_plot(self):
        """Generate and display the heatmap plot."""
        if not self.db:
            messagebox.showwarning("Warning", "No database loaded")
            return

        try:
            model_idx = int(self.heatmap_idx_var.get())
        except ValueError:
            messagebox.showerror("Error", "Model Index must be an integer.")
            return

        cmap_name = self.heatmap_cmap_var.get()
        try:
            colormap = plt.get_cmap(cmap_name)
        except:
            colormap = plt.cm.RdBu_r

        self.heatmap_fig.clear()
        try:
            self._update_status("Generating heatmap...")
            self.root.update()
            
            db_to_use = self.db.extract(self.db.all_keys(), **self.search_criteria) if self.search_criteria else self.db
            intrabunch_mode_heatmap(
                db=db_to_use,
                model_idx=model_idx,
                size=(10, 6),
                colormap=colormap)
            self.heatmap_canvas.draw()
            self._update_status("Heatmap generated")
        except Exception as e:
            messagebox.showerror("Error", f"Heatmap generation failed: {str(e)}")
            self._update_status("Error generating heatmap")

    # ==================== VERSUS PLOT TAB ====================

    def _create_versus_plot_tab(self):
        """Create the Versus Plot tab for database-wide plots."""
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="Versus Plot")
        
        control_frame = ttk.LabelFrame(tab, text="Plot Controls", padding="5")
        control_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        tab.columnconfigure(0, weight=1)
        
        axes_frame = ttk.Frame(control_frame)
        axes_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(axes_frame, text="X-axis:").pack(side=tk.LEFT)
        self.versus_x_var = tk.StringVar()
        self.versus_x_combo = ttk.Combobox(axes_frame, textvariable=self.versus_x_var, state="readonly", width=20)
        self.versus_x_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(axes_frame, text="Y-axis:").pack(side=tk.LEFT)
        self.versus_y_var = tk.StringVar()
        self.versus_y_combo = ttk.Combobox(axes_frame, textvariable=self.versus_y_var, state="readonly", width=20)
        self.versus_y_combo.pack(side=tk.LEFT, padx=5)
        
        color_frame = ttk.Frame(control_frame)
        color_frame.pack(fill=tk.X, pady=5)
        ttk.Label(color_frame, text="Color by:").pack(side=tk.LEFT)
        self.versus_color_var = tk.StringVar(value="None")
        self.versus_color_combo = ttk.Combobox(color_frame, textvariable=self.versus_color_var, state="readonly", width=20)
        self.versus_color_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Generate Plot", command=self._generate_versus_plot).pack(pady=10)
        
        plot_frame = ttk.LabelFrame(tab, text="Plot", padding="5")
        plot_frame.grid(row=1, column=0, sticky="nsew")
        tab.rowconfigure(1, weight=1)
        
        self.versus_fig = Figure(figsize=(8, 6), dpi=100)
        self.versus_canvas = FigureCanvasTkAgg(self.versus_fig, master=plot_frame)
        self.versus_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.versus_canvas, plot_frame).update()

    def _generate_versus_plot(self):
        """Generate a versus plot from filtered data."""
        if not self.db:
            messagebox.showwarning("Warning", "No database loaded")
            return
        
        x_prop, y_prop = self.versus_x_var.get(), self.versus_y_var.get()
        
        if not (x_prop and y_prop):
            messagebox.showwarning("Warning", "Please select both X and Y axes")
            return
        
        try:
            self.versus_fig.clear()
            ax = self.versus_fig.add_subplot(111)
            
            db_to_use = self.db.extract(self.db.all_keys(), **self.search_criteria) if self.search_criteria else self.db
            
            # Extract data
            x_data = []
            y_data = []
            for doc in db_to_use.db.all():
                if x_prop in doc and y_prop in doc:
                    x_val = doc[x_prop]
                    y_val = doc[y_prop]
                    if isinstance(x_val, (int, float, np.number)) and isinstance(y_val, (int, float, np.number)):
                        if np.isfinite(x_val) and np.isfinite(y_val):
                            x_data.append(x_val)
                            y_data.append(y_val)
            
            if x_data and y_data:
                ax.scatter(x_data, y_data, alpha=0.6, s=50)
                ax.set_xlabel(x_prop, fontsize=12)
                ax.set_ylabel(y_prop, fontsize=12)
                ax.grid(True, alpha=0.3)
                self.versus_fig.tight_layout()
                self.versus_canvas.draw()
                self._update_status(f"Versus plot generated: {y_prop} vs {x_prop}")
            else:
                messagebox.showwarning("Warning", "No valid data points found for the selected properties")
        except Exception as e:
            messagebox.showerror("Error", f"Plot generation failed: {str(e)}")

    # ==================== HISTOGRAM TAB ====================

    def _create_histogram_tab(self):
        """Create the Histogram tab."""
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="Histogram")
        
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
        """Generate a histogram from filtered data."""
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
            
            # Extract data
            data = []
            for doc in db_to_use.db.all():
                if prop in doc:
                    val = doc[prop]
                    if isinstance(val, (int, float, np.number)) and np.isfinite(val):
                        data.append(val)
            
            if data:
                ax.hist(data, bins=bins_val if isinstance(bins_val, int) else 'auto')
                if self.hist_log_var.get():
                    ax.set_yscale('log')
                ax.set_xlabel(prop, fontsize=12)
                ax.set_ylabel("Count", fontsize=12)
                ax.grid(True, alpha=0.3, axis='y')
                self.hist_fig.tight_layout()
                self.hist_canvas.draw()
                self._update_status(f"Histogram generated: {prop}")
            else:
                messagebox.showwarning("Warning", "No valid data found for the selected property")
        except Exception as e:
            messagebox.showerror("Error", f"Histogram generation failed: {str(e)}")

    # ==================== INDIVIDUAL PLOT TAB ====================

    def _create_individual_plot_tab(self):
        """Create the Individual Simulation Plot tab."""
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
        if len(self.individual_columns) > 0:
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
        
        info_frame = ttk.LabelFrame(right_frame, text="Simulation Info", padding="5")
        info_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        self.individual_info_text = scrolledtext.ScrolledText(info_frame, height=6, wrap=tk.WORD)
        self.individual_info_text.pack(fill=tk.BOTH, expand=True)
        
        plot_frame = ttk.LabelFrame(right_frame, text="Simulation Evolution", padding="5")
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
        """Update the individual simulation list from filtered results."""
        for item in self.individual_sim_tree.get_children():
            self.individual_sim_tree.delete(item)
        
        if not self.db:
            return
        
        self.individual_sim_tree.configure(columns=self.individual_columns)
        for col in self.individual_columns:
            self.individual_sim_tree.heading(col, text=col)
            self.individual_sim_tree.column(col, width=120 if col != "doc_id" else 60)
        
        # Only show simulations that match current filter
        simulations = self.db.where(**self.search_criteria) if self.search_criteria else self.db.where()
        for sim in simulations:
            values = [sim.doc_id if col == "doc_id" else self._format_value(sim.get(col, "N/A")) 
                     for col in self.individual_columns]
            self.individual_sim_tree.insert("", tk.END, values=values)

    def _plot_individual_simulation(self, *args):
        """Plot selected simulation evolution data."""
        selections = self.individual_sim_tree.selection()
        if not (selections and self.db):
            self.individual_info_text.delete(1.0, tk.END)
            self.individual_fig.clear()
            self.individual_canvas.draw()
            return
        
        try:
            doc_id_idx = self.individual_columns.index("doc_id")
            doc_id = int(self.individual_sim_tree.item(selections[0])['values'][doc_id_idx])
        except (ValueError, IndexError):
            return
        
        model = InstabilityModel(self.db.db, doc_id)
        
        # Display simulation info
        self.individual_info_text.delete(1.0, tk.END)
        info_lines = [
            f"Doc ID: {model.doc_id}",
            f"Path: {model.path}",
            f"N Turns: {model.n_turns}",
            f"N Slices: {model.n_slices}",
            f"Growth Rate (Centroid): {self._format_value(model.growth_rate_centroid)}",
            f"Dominant Mode Index: {model.dominant_mode_idx}",
            f"Blowup Turn First: {self._format_value(model.blowup_turn_first)}",
        ]
        self.individual_info_text.insert(tk.END, "\n".join(info_lines))
        
        # Plot evolution data
        self.individual_fig.clear()
        try:
            # Create subplots for different observables
            n_plots = 3
            for i, (data_attr, label) in enumerate([
                ('mean_x', 'Centroid X'),
                ('epsn_x', 'Norm. Emittance X'),
                ('sigma_x', 'Bunch Length X')
            ]):
                ax = self.individual_fig.add_subplot(n_plots, 1, i + 1)
                data = getattr(model, data_attr, np.array([]))
                
                if len(data) > 0:
                    turns = np.arange(len(data))
                    valid = np.isfinite(data)
                    if np.any(valid):
                        ax.plot(turns[valid], data[valid], 'b-', linewidth=1.5)
                        ax.grid(True, alpha=0.3)
                        ax.set_ylabel(label, fontsize=10)
                        if i == n_plots - 1:
                            ax.set_xlabel("Turns", fontsize=10)
                else:
                    ax.text(0.5, 0.5, f"No data for {label}", ha='center', va='center', transform=ax.transAxes)
            
            self.individual_fig.tight_layout()
            self.individual_canvas.draw()
            self._update_status(f"Plotted simulation {doc_id}")
        except Exception as e:
            self.individual_info_text.insert(tk.END, f"\n\nError plotting: {str(e)}")
            self._update_status(f"Error plotting simulation: {str(e)}")

    # ==================== POPULATION HELPERS ====================

    def _populate_plot_options(self):
        """Populate plot option dropdowns with available properties."""
        if not self.db:
            return
        
        plotable_props = self._get_plotable_properties()
        
        # Update growth rate dropdowns
        if hasattr(self, 'growth_density_combo'):
            self.growth_density_combo['values'] = plotable_props
            if "init_unif_edens_dip" in plotable_props:
                self.growth_density_var.set("init_unif_edens_dip")
            elif len(plotable_props) > 0:
                self.growth_density_var.set(plotable_props[0])
        
        if hasattr(self, 'growth_rate_combo'):
            self.growth_rate_combo['values'] = plotable_props
            if "growth_rate_mode" in plotable_props:
                self.growth_rate_var.set("growth_rate_mode")
            elif "growth_rate_centroid" in plotable_props:
                self.growth_rate_var.set("growth_rate_centroid")
            elif len(plotable_props) > 0:
                self.growth_rate_var.set(plotable_props[0])
        
        # Update blowup dropdowns
        if hasattr(self, 'blowup_strength_combo'):
            self.blowup_strength_combo['values'] = plotable_props
            if "strength_factor" in plotable_props:
                self.blowup_strength_var.set("strength_factor")
            elif len(plotable_props) > 0:
                self.blowup_strength_var.set(plotable_props[0])
        
        if hasattr(self, 'blowup_turns_combo'):
            self.blowup_turns_combo['values'] = plotable_props
            if "blowup_turn_first" in plotable_props:
                self.blowup_turns_var.set("blowup_turn_first")
            elif len(plotable_props) > 0:
                self.blowup_turns_var.set(plotable_props[0])
        
        # Update versus and histogram dropdowns
        if hasattr(self, 'versus_x_combo'):
            self.versus_x_combo['values'] = plotable_props
        if hasattr(self, 'versus_y_combo'):
            self.versus_y_combo['values'] = plotable_props
        if hasattr(self, 'versus_color_combo'):
            self.versus_color_combo['values'] = ["None"] + plotable_props
        if hasattr(self, 'hist_prop_combo'):
            self.hist_prop_combo['values'] = plotable_props

    def _get_plotable_properties(self) -> List[str]:
        """Get list of plotable properties from database."""
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
        
        return sorted(plotable_props)


def main():
    """Main entry point for the application."""
    root = tk.Tk()
    app = InstabApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
