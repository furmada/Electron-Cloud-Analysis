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
from typing import Optional, List, Dict, Any, Tuple
import threading
from collections import defaultdict
import numpy as np

# Import your ECA modules
try:
    from eca import SimDB, ECModel, InstabilityModel, FurmanNoPhotoFit, FurmanPhotoFit, FurmanNPMCFit, Fit, WhereIn
    from ecaplots import model_plot, versus_plot, histogram_plot
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
except ImportError as e:
    print(f"Error importing ECA modules: {e}")
    print("Make sure eca.py and ecaplots.py are in the same directory or PYTHONPATH")
    sys.exit(1)

class ECAApp:
    """Main application class for the Electron Cloud Analysis GUI."""
    
    def __init__(self, root: tk.Tk, db: Optional[SimDB] = None, is_temp: bool = False):
        self.root = root
        self.is_temp = is_temp
        
        title_prefix = "[TEMP WINDOW] " if is_temp else ""
        self.root.title(f"{title_prefix}ECA GUI - Electron Cloud Analysis")
        self.root.geometry("1400x900")
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # Setup application-wide fonts first
        self._setup_fonts()
        
        # Application state
        self.db: Optional[SimDB] = db
        self.search_criteria: Dict[str, Any] = {}
        self.current_models: List[ECModel] = []
        self.selected_simulations: List[str] = []
        
        # Setup UI
        self._setup_ui()
        
        # If initialized with an existing DB (e.g. Temp Window), populate UI
        if self.db:
            self._update_overview_tab()
            self._populate_filter_options()
            self._populate_plot_options()
            self._populate_individual_options()
            self._update_individual_sim_list()
            
            label_text = "Loaded: Extracted Temp DB" if self.is_temp else "Loaded Database"
            self.info_label.config(text=label_text)
            self._update_status(f"Loaded {len(self.db.where())} simulations")

    def _on_closing(self):
        """Ensure the process terminates fully for main window, or just destroys temp window."""
        window_type = "temporary " if self.is_temp else ""
        if messagebox.askokcancel("Quit", f"Do you want to close this {window_type}window?"):
            if self.is_temp:
                self.root.destroy()
            else:
                self.root.destroy()
                sys.exit(0)

    def _populate_filter_options(self):
        """Populate the filter property dropdown with available properties."""
        if not self.db:
            return
            
        all_keys = self.db.all_keys()
        total_count = len(self.db.where())
        
        valid_properties = []
        for key in all_keys:
            if key == 'path':
                continue
            unique_values = self.db.unique(key)
            if len(unique_values) > 1 and len(unique_values) < total_count:
                valid_properties.append(key)
                
        self.filter_property_combo['values'] = valid_properties

    def _on_property_select(self, event=None):
        """Populate the values listbox when a property is selected."""
        if not self.db:
            return
            
        prop = self.filter_property_var.get()
        if not prop:
            return
            
        # Get all unique values for this property from the database
        all_values = set()
        for doc in self.db.db.all():
            if prop in doc:
                val = doc[prop]
                # TinyDB might return unhashable types like lists; handle them
                if isinstance(val, list):
                    all_values.add(tuple(val))
                else:
                    all_values.add(val)
        
        # Sort and store the mapping (string representation -> actual data)
        sorted_vals = sorted(list(all_values), key=lambda x: str(x))
        self.filter_val_map = {self._format_value(v): v for v in sorted_vals}
        
        # Update the Listbox display
        self.filter_values_listbox.delete(0, tk.END)
        for val_str in self.filter_val_map.keys():
            self.filter_values_listbox.insert(tk.END, val_str)

    def _format_value(self, val: Any) -> str:
        """Format numbers consistently for display."""
        if isinstance(val, bool):
            return str(val)  # Prevent booleans from acting as 1/0
            
        if isinstance(val, (int, float, np.number)):
            if np.isnan(val) or np.isinf(val):
                return str(val)
                
            # Scientific notation for magnitudes >= 1000
            if abs(val) >= 1000 or abs(val) < 1e-4:
                return f"{val:.3E}"
            # 3 decimal places for floats < 1000
            elif isinstance(val, (float, np.floating)):
                return f"{val:.4f}"
            else:
                return str(val) # Keep standard integers as standard integers
                
        return str(val)
        
    def _setup_fonts(self):
        """Increase base font size to match standard system sizes."""
        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(size=11)
        text_font = tkfont.nametofont("TkTextFont")
        text_font.configure(size=11)
        fixed_font = tkfont.nametofont("TkFixedFont")
        fixed_font.configure(size=11)
        
        # Apply to TTK styles
        style = ttk.Style()
        style.configure(".", font="TkDefaultFont")
        style.configure("Treeview.Heading", font="TkDefaultFont", weight="bold")
        
    def _setup_ui(self):
        """Setup the main user interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Top toolbar
        self._create_toolbar(main_frame)
        
        # Main content area with tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=1, column=0, sticky="nsew", pady=5)
        
        # Create tabs
        self._create_overview_tab()
        self._create_filter_tab()
        self._create_fitting_tab()
        self._create_plotting_tab()
        self._create_histogram_tab()
        self._create_individual_plot_tab()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=2, column=0, sticky="ew")
        
    def _create_toolbar(self, parent):
        """Create the top toolbar with file operations."""
        toolbar = ttk.Frame(parent)
        toolbar.grid(row=0, column=0, sticky="ew", pady=5)
        
        # Load database button
        load_btn = ttk.Button(toolbar, text="Load Database", command=self._load_database)
        load_btn.pack(side=tk.LEFT, padx=2)
        
        # Clear database button
        clear_btn = ttk.Button(toolbar, text="Clear Database", command=self._clear_database)
        clear_btn.pack(side=tk.LEFT, padx=2)
        
        # Save Database Button (Only for Temp Windows)
        if self.is_temp:
            save_btn = ttk.Button(toolbar, text="Save Database As...", command=self._save_database)
            save_btn.pack(side=tk.LEFT, padx=2)
        
        # Spacer
        ttk.Label(toolbar, text="").pack(side=tk.LEFT, expand=True)
        
        # Info label
        self.info_label = ttk.Label(toolbar, text="No database loaded", font=("TkDefaultFont", 11, "bold"))
        self.info_label.pack(side=tk.RIGHT, padx=5)
        
    def _load_database(self):
        """Open file dialog and load a database."""
        filename = filedialog.askopenfilename(
            title="Select Database File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            self.status_var.set(f"Loading database from {filename}...")
            self.root.update()
            
            try:
                self.db = SimDB(filename, verbose=True)
                
                # Path handling
                simulations = self.db.where()
                status_suffix = ""
                if simulations:
                    first_path = simulations[0].get('path', '')
                    if first_path.startswith('.'):
                        db_dir = os.path.dirname(os.path.abspath(filename))
                        os.chdir(db_dir)
                        status_suffix = f" (CWD set to {db_dir})"
                
                self.search_criteria = {}
                self.active_filters = {}
                self.current_models = []
                self.selected_simulations = []
                
                # Update UI elements across all tabs
                self._update_overview_tab()
                self._populate_filter_options()
                self._populate_plot_options()
                self._populate_individual_options()
                self._update_individual_sim_list()
                
                # Update state displays
                self.info_label.config(text=f"Loaded: {os.path.basename(filename)}")
                self._update_status(f"Loaded {len(self.db.db)} simulations{status_suffix}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load database: {str(e)}")
                self._update_status("Error loading database")
                
    def _save_database(self):
        """Save the in-memory/temp database to a physical JSON file."""
        if not self.db:
            return
            
        filename = filedialog.asksaveasfilename(
            title="Save Database As",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                from tinydb import TinyDB
                from tinydb.storages import JSONStorage
                from tinydb.middlewares import CachingMiddleware
                
                # Create a fresh file and insert all documents
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
            self.search_criteria = {}
            self.active_filters = {}
            self.current_models = []
            self.selected_simulations = []
            self._clear_all_tabs()
            self.info_label.config(text="No database loaded")
            self._update_status("Database cleared")
            
    def _update_status(self, message: str):
        """Update the status bar."""
        self.status_var.set(message)
        self.root.update_idletasks()
        
    def _clear_all_tabs(self):
        """Clear all tab contents."""
        self._clear_overview_tab()
        self._clear_filter_tab()
        self._clear_fitting_tab()
        self._clear_plotting_tab()
        self._clear_histogram_tab()
        self._clear_individual_plot_tab()
        
    # ==================== OVERVIEW TAB ====================
    
    def _create_overview_tab(self):
        """Create the Overview tab."""
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="Overview")
        
        # Left panel - Simulation list
        left_frame = ttk.LabelFrame(tab, text="Simulations", padding="5")
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)
        
        # Treeview for simulations
        columns = ("Path",)
        self.sim_tree = ttk.Treeview(left_frame, columns=columns, show="headings", height=20)
        self.sim_tree.heading("Path", text="Simulation Path")
        self.sim_tree.column("Path", width=500)
        
        sim_scroll = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.sim_tree.yview)
        self.sim_tree.configure(yscrollcommand=sim_scroll.set)
        
        self.sim_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sim_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Right panel - Properties summary
        right_frame = ttk.LabelFrame(tab, text="Properties Summary", padding="5")
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        tab.columnconfigure(1, weight=1)
        
        self.prop_text = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD, height=20)
        self.prop_text.pack(fill=tk.BOTH, expand=True)
        
        # Refresh button
        refresh_btn = ttk.Button(tab, text="Refresh Overview", command=self._update_overview_tab)
        refresh_btn.grid(row=1, column=0, columnspan=2, pady=5)
        
    def _update_overview_tab(self):
        """Update the overview tab with current database information (respects filters)."""
        if not self.db:
            return
            
        # Clear previous data
        for item in self.sim_tree.get_children():
            self.sim_tree.delete(item)
        self.prop_text.delete(1.0, tk.END)
        
        # Get simulations based on current filter criteria
        simulations = self.db.where(**self.search_criteria)
        total_count = len(simulations)
        
        # Populate simulation list with missing path detection
        for sim in simulations:
            path = sim.get('path', 'Unknown')
            display_path = path
            if not os.path.exists(path):
                display_path = "(Path not found) " + display_path
            self.sim_tree.insert("", tk.END, values=(display_path,))
            
        # Calculate property statistics inside the filtered set
        all_keys = self.db.all_keys()
        property_stats = {}
        
        for key in all_keys:
            if key == 'path':  # Skip path as it's per-simulation
                continue
            unique_values = self.db.unique(key, **self.search_criteria)
            if len(unique_values) > 1 and len(unique_values) < total_count:
                property_stats[key] = {
                    'unique_count': len(unique_values),
                    'values': unique_values
                }
                
        # Display property summary
        self.prop_text.insert(tk.END, f"Total Displayed Simulations: {total_count}\n")
        self.prop_text.insert(tk.END, "=" * 50 + "\n\n")
        self.prop_text.insert(tk.END, f"Properties with multiple values ({len(property_stats)}):\n\n")
        
        for prop, stats in sorted(property_stats.items()):
            self.prop_text.insert(tk.END, f"{prop}: {stats['unique_count']} unique values\n")
            if stats['unique_count'] < 100:
                formatted_vals = [self._format_value(v) for v in stats['values']]
                self.prop_text.insert(tk.END, f"  Values: {', '.join(formatted_vals)}\n")
            else:
                formatted_vals = [self._format_value(v) for v in stats['values'][:10]]
                self.prop_text.insert(tk.END, f"  Sample: {', '.join(formatted_vals)}\n")
            self.prop_text.insert(tk.END, "\n")
            
        self._update_status(f"Overview updated: {total_count} simulations")
        
    def _clear_overview_tab(self):
        """Clear the overview tab."""
        for item in self.sim_tree.get_children():
            self.sim_tree.delete(item)
        self.prop_text.delete(1.0, tk.END)
        
    # ==================== FILTER TAB ====================
    
    def _create_filter_tab(self):
        """Create the Filter tab with dynamic filtering support."""
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="Filter")
        
        # Mode Selection
        mode_frame = ttk.Frame(tab)
        mode_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        ttk.Label(mode_frame, text="Filter Mode:").pack(side=tk.LEFT, padx=(0, 10))
        self.filter_mode_var = tk.StringVar(value="exact")
        ttk.Radiobutton(mode_frame, text="Exact Match", variable=self.filter_mode_var, value="exact", command=self._toggle_filter_mode).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_frame, text="Condition (>, <, ==)", variable=self.filter_mode_var, value="condition", command=self._toggle_filter_mode).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_frame, text="Custom Expression", variable=self.filter_mode_var, value="expression", command=self._toggle_filter_mode).pack(side=tk.LEFT, padx=5)
        
        # Filter controls
        control_frame = ttk.LabelFrame(tab, text="Filter Criteria", padding="5")
        control_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        tab.columnconfigure(0, weight=1)
        
        # Top part of control frame: Property selection (Used by exact and condition)
        self.prop_frame = ttk.Frame(control_frame)
        self.prop_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(self.prop_frame, text="Property:").pack(side=tk.LEFT)
        self.filter_property_var = tk.StringVar()
        self.filter_property_combo = ttk.Combobox(self.prop_frame, textvariable=self.filter_property_var, state="readonly", width=30)
        self.filter_property_combo.pack(side=tk.LEFT, padx=5)
        self.filter_property_combo.bind("<<ComboboxSelected>>", self._on_property_select)
        
        # Container for dynamic inputs
        self.input_container = ttk.Frame(control_frame)
        self.input_container.pack(fill=tk.X, pady=5)
        
        # --- Exact Match UI ---
        self.exact_frame = ttk.Frame(self.input_container)
        ttk.Label(self.exact_frame, text="Values:").pack(side=tk.LEFT)
        self.filter_values_listbox = tk.Listbox(self.exact_frame, selectmode=tk.MULTIPLE, height=5, width=50)
        self.filter_values_listbox.pack(side=tk.LEFT, padx=5)
        value_scroll = ttk.Scrollbar(self.exact_frame, orient=tk.VERTICAL, command=self.filter_values_listbox.yview)
        self.filter_values_listbox.configure(yscrollcommand=value_scroll.set)
        value_scroll.pack(side=tk.LEFT, fill=tk.Y)
        
        # --- Condition UI ---
        self.condition_frame = ttk.Frame(self.input_container)
        ttk.Label(self.condition_frame, text="Operator:").pack(side=tk.LEFT)
        self.cond_op_var = tk.StringVar(value=">")
        self.cond_op_combo = ttk.Combobox(self.condition_frame, textvariable=self.cond_op_var, state="readonly", width=5, values=[">", "<", ">=", "<=", "==", "!="])
        self.cond_op_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(self.condition_frame, text="Value:").pack(side=tk.LEFT, padx=(10, 0))
        self.cond_val_var = tk.StringVar()
        ttk.Entry(self.condition_frame, textvariable=self.cond_val_var, width=20).pack(side=tk.LEFT, padx=5)
        
        # --- Expression UI ---
        self.expression_frame = ttk.Frame(self.input_container)
        ttk.Label(self.expression_frame, text="Expression:").pack(side=tk.LEFT)
        self.expr_var = tk.StringVar()
        ttk.Entry(self.expression_frame, textvariable=self.expr_var, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Label(self.expression_frame, text="(e.g., Ne_0 > 1e10 and buildup == True)").pack(side=tk.LEFT, padx=5)
        
        # Add filter button
        add_filter_btn = ttk.Button(control_frame, text="Add Filter", command=self._add_filter)
        add_filter_btn.pack(pady=5)
        
        # Active filters display
        filters_frame = ttk.LabelFrame(tab, text="Active Filters", padding="5")
        filters_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 10))
        tab.rowconfigure(2, weight=1)
        
        self.active_filters_text = scrolledtext.ScrolledText(filters_frame, height=8, wrap=tk.WORD)
        self.active_filters_text.pack(fill=tk.BOTH, expand=True)
        
        # Action buttons
        button_frame = ttk.Frame(tab)
        button_frame.grid(row=3, column=0, pady=10)
        
        apply_filter_btn = ttk.Button(button_frame, text="Apply Filter", command=self._apply_filter)
        apply_filter_btn.pack(side=tk.LEFT, padx=5)
        
        clear_filters_btn = ttk.Button(button_frame, text="Clear All Filters", command=self._clear_filters)
        clear_filters_btn.pack(side=tk.LEFT, padx=5)
        
        reset_btn = ttk.Button(button_frame, text="Reset to Full Database", command=self._reset_filter)
        reset_btn.pack(side=tk.LEFT, padx=5)
        
        temp_window_btn = ttk.Button(button_frame, text="New Temp. Window", command=self._open_temp_window)
        temp_window_btn.pack(side=tk.LEFT, padx=5)
        
        # Store active filters locally as a list to support multiple rules
        self.active_filters_list: List[Dict[str, Any]] = []
        self.filter_val_map = {}
        
        # Initialize UI state
        self._toggle_filter_mode()

    def _toggle_filter_mode(self, *args):
        """Toggle the visibility of input frames based on selected mode."""
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
        elif mode == "expression":
            self.prop_frame.pack_forget() # Not needed for general expression
            self.expression_frame.pack(fill=tk.X)

    def _add_filter(self):
        """Add a filter criterion based on current mode."""
        mode = self.filter_mode_var.get()
        
        if mode == "exact":
            prop = self.filter_property_var.get()
            if not prop:
                messagebox.showwarning("Warning", "Please select a property to filter on")
                return
                
            selected_indices = self.filter_values_listbox.curselection()
            if not selected_indices:
                messagebox.showwarning("Warning", "Please select at least one value")
                return
                
            selected_strings = [self.filter_values_listbox.get(i) for i in selected_indices]
            exact_values = [self.filter_val_map[val_str] for val_str in selected_strings]
            
            # Group exact filters by property
            found = False
            for f in self.active_filters_list:
                if f["type"] == "exact" and f["property"] == prop:
                    f["values"].extend(exact_values)
                    f["values"] = list(set(f["values"]))
                    found = True
                    break
            if not found:
                self.active_filters_list.append({"type": "exact", "property": prop, "values": exact_values})
                
        elif mode == "condition":
            prop = self.filter_property_var.get()
            if not prop:
                messagebox.showwarning("Warning", "Please select a property for the condition")
                return
            
            op = self.cond_op_var.get()
            val_str = self.cond_val_var.get().strip()
            if not val_str:
                messagebox.showwarning("Warning", "Please provide a value for the condition")
                return
                
            try:
                # Try to parse as a numeric
                val = float(val_str)
                if val.is_integer(): val = int(val)
            except ValueError:
                val = val_str # Fallback to string matching
                
            self.active_filters_list.append({"type": "condition", "property": prop, "operator": op, "value": val})
            
        elif mode == "expression":
            expr = self.expr_var.get().strip()
            if not expr:
                messagebox.showwarning("Warning", "Please enter an expression")
                return
                
            self.active_filters_list.append({"type": "expression", "expr": expr})
            
        self._update_active_filters_display()
        
    def _update_active_filters_display(self):
        """Update the active filters display text box."""
        self.active_filters_text.delete(1.0, tk.END)
        if not self.active_filters_list:
            self.active_filters_text.insert(tk.END, "No active filters\n")
            return
            
        self.active_filters_text.insert(tk.END, "Active Filters:\n")
        self.active_filters_text.insert(tk.END, "-" * 30 + "\n")
        
        for i, f_data in enumerate(self.active_filters_list):
            if f_data["type"] == "exact":
                formatted_vals = [self._format_value(v) for v in f_data["values"]]
                self.active_filters_text.insert(tk.END, f"[{i+1}] Exact: {f_data['property']} in ({', '.join(formatted_vals)})\n")
            elif f_data["type"] == "condition":
                val_fmt = f"'{f_data['value']}'" if isinstance(f_data['value'], str) else f_data['value']
                self.active_filters_text.insert(tk.END, f"[{i+1}] Condition: {f_data['property']} {f_data['operator']} {val_fmt}\n")
            elif f_data["type"] == "expression":
                self.active_filters_text.insert(tk.END, f"[{i+1}] Expression: {f_data['expr']}\n")
                
    def _build_search_criteria(self):
        """Build the query dictionary from active filters, compatible with eca.py's where()."""
        self.search_criteria = {}
        for i, f_data in enumerate(self.active_filters_list):
            if f_data["type"] == "exact":
                self.search_criteria[f_data["property"]] = WhereIn(*f_data["values"])
                
            elif f_data["type"] == "condition":
                prop = f_data["property"]
                op = f_data["operator"]
                val = f_data["value"]
                
                # Factory isolated to prevent late-binding Python loop issues
                def build_callable(p, o, v):
                    def cond_fn(result):
                        if p not in result: return False
                        res_val = result[p]
                        try:
                            if o == ">": return res_val > v
                            if o == "<": return res_val < v
                            if o == ">=": return res_val >= v
                            if o == "<=": return res_val <= v
                            if o == "==": return res_val == v
                            if o == "!=": return res_val != v
                        except TypeError:
                            return False # Usually handles string/int comparison failures
                        return False
                    return cond_fn
                
                # Assign with a leading underscore so ECA evaluates it as a Callable(result)
                self.search_criteria[f"_cond_{i}"] = build_callable(prop, op, val)
                
            elif f_data["type"] == "expression":
                # Using a leading underscore makes ECA run eval() natively inside the result scope
                self.search_criteria[f"_expr_{i}"] = f_data["expr"]

    def _apply_filter(self):
        """Update the global search criteria used across the application."""
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
        
    def _clear_filter_tab(self):
        """Clear the filter tab entirely."""
        self.search_criteria.clear()
        self.active_filters_list.clear()
        self._update_active_filters_display()
        self.filter_property_var.set("")
        self.filter_values_listbox.delete(0, tk.END)
        self.cond_val_var.set("")
        self.expr_var.set("")
        
    def _reset_filter(self):
        """Reset the search criteria and reload the full database."""
        if not self.db:
            return
            
        self._clear_filters()
        self.search_criteria = {}
        
        # Reload all models from the full database
        self.current_models = [ECModel(self.db.db, doc) for doc in self.db.db.all()]
        
        # Refresh the UI
        self._update_overview_tab()
        self._update_individual_sim_list()
        
        self._update_status("Database reset to full dataset.")
        messagebox.showinfo("Reset", "Filters cleared. Showing all simulations.")

    def _open_temp_window(self):
        """Open a new GUI window containing only the currently filtered results."""
        if not self.current_models:
            messagebox.showwarning("Warning", "No simulations to display in a new window.")
            return
            
        # Create a new top-level window
        new_window = tk.Toplevel(self.root)
        
        # Initialize a new instance of ECAApp in the new window
        # We pass the current db and set is_temp=True
        new_app = ECAApp(new_window, db=self.db, is_temp=True)
        
        # Override the new app's models with our currently filtered subset
        new_app.current_models = self.current_models.copy()
        
        # Update the new window's UI to reflect the subset
        new_app._update_overview_tab()
        new_app._update_individual_sim_list()
        new_app._update_status(f"Initialized with {len(self.current_models)} simulations.")

    # ==================== FITTING TAB ====================
    
    def _create_fitting_tab(self):
        """Create the Fitting tab."""
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="Fitting")
        
        # Model selection
        model_frame = ttk.LabelFrame(tab, text="Fit Model", padding="5")
        model_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        tab.columnconfigure(0, weight=1)
        
        ttk.Label(model_frame, text="Select Fit Model:").pack(side=tk.LEFT)
        self.fit_model_var = tk.StringVar(value="FurmanNoPhoto")
        self.fit_model_combo = ttk.Combobox(model_frame, textvariable=self.fit_model_var, state="readonly", width=30)
        self.fit_model_combo['values'] = ["FurmanNoPhoto", "FurmanNPMC", "FurmanPhoto"]
        self.fit_model_combo.pack(side=tk.LEFT, padx=5)
        
        # Selection options
        selection_frame = ttk.LabelFrame(tab, text="Selection", padding="5")
        selection_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        tab.rowconfigure(1, weight=1)
        
        self.selection_mode = tk.StringVar(value="filtered")
        ttk.Radiobutton(selection_frame, text="Use Filtered Simulations", 
                       variable=self.selection_mode, value="filtered").pack(anchor=tk.W)
        ttk.Radiobutton(selection_frame, text="Use All Simulations", 
                       variable=self.selection_mode, value="all").pack(anchor=tk.W)
                       
        # Progress display
        progress_frame = ttk.LabelFrame(tab, text="Progress", padding="5")
        progress_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        
        self.progress_var = tk.StringVar(value="Ready")
        progress_bar = ttk.Label(progress_frame, textvariable=self.progress_var)
        progress_bar.pack(fill=tk.X)
        
        # Action buttons
        button_frame = ttk.Frame(tab)
        button_frame.grid(row=3, column=0, pady=10)
        
        fit_btn = ttk.Button(button_frame, text="Apply Fit", command=self._apply_fit)
        fit_btn.pack(side=tk.LEFT, padx=5)
        
        refit_btn = ttk.Button(button_frame, text="Refit All", command=self._refit_all)
        refit_btn.pack(side=tk.LEFT, padx=5)
        
        # Results display
        results_frame = ttk.LabelFrame(tab, text="Fitting Results", padding="5")
        results_frame.grid(row=4, column=0, sticky="nsew")
        tab.rowconfigure(4, weight=1)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=15, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
    def _get_models_to_fit(self) -> List[ECModel]:
        """Get the list of models to fit based on selection mode."""
        if not self.db:
            return []
            
        if self.selection_mode.get() == "filtered" and self.search_criteria:
            # Note: Because ECModel wraps self.db.db, fits WILL save to the master file
            return [ECModel(self.db.db, doc) for doc in self.db.where(**self.search_criteria)]
        else:
            return [ECModel(self.db.db, doc) for doc in self.db.where()]
            
    def _apply_fit(self, refit=False):
        """Apply the selected fit model to the chosen simulations."""
        if not self.db:
            messagebox.showwarning("Warning", "No database loaded")
            return
            
        model_name = self.fit_model_var.get()
        models = list(self._get_models_to_fit())
        
        if not models:
            messagebox.showwarning("Warning", "No simulations to fit")
            return
            
        # Create fit object
        #try:
        if model_name == "FurmanNoPhoto":
            fit = FurmanNoPhotoFit()
        elif model_name == "FurmanNPMC":
            fit = FurmanNPMCFit(self.db) # Injection here
        elif model_name == "FurmanPhoto":
            fit = FurmanPhotoFit()
            messagebox.showerror("Error", f"Unknown fit model: {model_name}")
            return
            
        # Apply fit to all models
        self.progress_var.set("Starting fit...")
        self.root.update()
        
        success_count = 0
        fail_count = 0
        
        for i in range(len(models)):
            result = fit.fit(models[i], refit=refit)
            models[i] = result is not None
            if result is not None:
                success_count += 1
            else:
                fail_count += 1
                
            self.progress_var.set(f"Fitting: {i+1}/{len(models)} ({success_count} success, {fail_count} failed)")
            self.root.update()
            
        # Display results
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Fitting Complete\n")
        self.results_text.insert(tk.END, "=" * 50 + "\n")
        self.results_text.insert(tk.END, f"Total: {len(models)}\n")
        self.results_text.insert(tk.END, f"Successful: {success_count}\n")
        self.results_text.insert(tk.END, f"Failed: {fail_count}\n\n")
        
        self._update_status(f"Fitting complete: {success_count}/{len(models)} successful")
        
        # except Exception as e:
        #     messagebox.showerror("Error", f"Fitting failed: {str(e)}")
        #     self._update_status("Fitting failed")
            
    def _refit_all(self):
        """Refit all simulations with the selected model."""
        if not self.db:
            messagebox.showwarning("Warning", "No database loaded")
            return
            
        if not messagebox.askyesno("Confirm", "This will refit all selected simulations. Continue?"):
            return
            
        self._apply_fit(True)
        
    def _clear_fitting_tab(self):
        """Clear the fitting tab."""
        self.results_text.delete(1.0, tk.END)
        self.progress_var.set("Ready")
        
    # ==================== PLOTTING TAB ====================
    
    def _create_plotting_tab(self):
        """Create the Plotting tab for database-wide plots."""
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="Database Plots")
        
        # Control panel
        control_frame = ttk.LabelFrame(tab, text="Plot Controls", padding="5")
        control_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        tab.columnconfigure(0, weight=1)
        
        # X and Y axis selection
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
        
        # Color by option
        color_frame = ttk.Frame(control_frame)
        color_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(color_frame, text="Color by:").pack(side=tk.LEFT)
        self.plot_color_var = tk.StringVar(value="None")
        self.plot_color_combo = ttk.Combobox(color_frame, textvariable=self.plot_color_var, state="readonly", width=20)
        self.plot_color_combo.pack(side=tk.LEFT, padx=5)
        
        # Plot button
        plot_btn = ttk.Button(control_frame, text="Generate Plot", command=self._generate_plot)
        plot_btn.pack(pady=10)
        
        # Plot display area
        plot_frame = ttk.LabelFrame(tab, text="Plot", padding="5")
        plot_frame.grid(row=1, column=0, sticky="nsew")
        tab.rowconfigure(1, weight=1)
        
        # Matplotlib figure
        self.plot_fig = Figure(figsize=(8, 6), dpi=100)
        self.plot_canvas = FigureCanvasTkAgg(self.plot_fig, master=plot_frame)
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Toolbar
        toolbar = NavigationToolbar2Tk(self.plot_canvas, plot_frame)
        toolbar.update()
        
    def _populate_plot_options(self):
        """Populate the plot axis dropdowns with available properties."""
        if not self.db:
            return
            
        all_keys = self.db.all_keys()
        
        # Filter for numeric properties suitable for plotting
        plotable_props = []
        
        for key in all_keys:
            if key in ['path', 'processed']:
                continue
                
            try:
                unique_values = self.db.unique(key)
                if len(unique_values) <= 1:
                    continue
                
                has_numeric = False
                for val in unique_values:
                    if isinstance(val, (int, float, np.number)):
                        if not (np.isnan(val) or np.isinf(val)):
                            has_numeric = True
                            break
                    elif isinstance(val, str):
                        try:
                            float_val = float(val)
                            if not (np.isnan(float_val) or np.isinf(float_val)):
                                has_numeric = True
                                break
                        except (ValueError, TypeError):
                            continue
                
                if has_numeric:
                    plotable_props.append(key)
                    
            except Exception as e:
                print(f"Skipping property {key}: {e}")
                continue
                
        self.plot_x_combo['values'] = plotable_props
        self.plot_y_combo['values'] = plotable_props
        self.plot_color_combo['values'] = ["None"] + plotable_props
        if hasattr(self, 'hist_prop_combo'):
            self.hist_prop_combo['values'] = plotable_props
        
    def _generate_plot(self):
        """Generate a plot using the ecaplots.versus_plot utility."""
        if not self.db:
            messagebox.showwarning("Warning", "No database loaded")
            return
            
        x_prop = self.plot_x_var.get()
        y_prop = self.plot_y_var.get()
        color_prop = self.plot_color_var.get() if self.plot_color_var.get() != "None" else None
        
        if not x_prop or not y_prop:
            messagebox.showwarning("Warning", "Please select both X and Y axes")
            return
            
        try:
            # Clear the existing figure
            self.plot_fig.clear()
            ax = self.plot_fig.add_subplot(111)
            
            # Since ecaplots.versus_plot generally expects a SimDB input, 
            # we temporarily extract a subset purely for read-only visualization purposes
            if self.search_criteria:
                db_to_use = self.db.extract(self.db.all_keys(), **self.search_criteria)
            else:
                db_to_use = self.db
            
            if color_prop:
                versus_plot(db_to_use, x_prop, y_prop, colorBy=color_prop, size=ax)
            else:
                versus_plot(db_to_use, x_prop, y_prop, size=ax)
            
            # Update the canvas to reflect changes
            self.plot_canvas.draw()
            
            title_suffix = f" (colored by {color_prop})" if color_prop else ""
            self._update_status(f"Plot generated: {y_prop} vs {x_prop}{title_suffix}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Plot generation failed: {str(e)}")
            self._update_status("Plot generation failed")
            
    def _clear_plotting_tab(self):
        """Clear the plotting tab."""
        self.plot_fig.clear()
        self.plot_canvas.draw()
        self.plot_x_var.set("")
        self.plot_y_var.set("")
        self.plot_color_var.set("None")
    
    # ==================== HISTOGRAM TAB ====================

    def _create_histogram_tab(self):
        """Create the Histogram tab for database-wide distributions."""
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="Histogram Plots")
        
        # Control panel
        control_frame = ttk.LabelFrame(tab, text="Plot Controls", padding="5")
        control_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        tab.columnconfigure(0, weight=1)
        
        # Settings frame
        settings_frame = ttk.Frame(control_frame)
        settings_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(settings_frame, text="Property:").pack(side=tk.LEFT)
        self.hist_prop_var = tk.StringVar()
        self.hist_prop_combo = ttk.Combobox(settings_frame, textvariable=self.hist_prop_var, state="readonly", width=20)
        self.hist_prop_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(settings_frame, text="Bins:").pack(side=tk.LEFT, padx=(10, 0))
        self.hist_bins_var = tk.StringVar(value="auto")
        self.hist_bins_entry = ttk.Entry(settings_frame, textvariable=self.hist_bins_var, width=10)
        self.hist_bins_entry.pack(side=tk.LEFT, padx=5)
        
        self.hist_log_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(settings_frame, text="Log Y-axis", variable=self.hist_log_var).pack(side=tk.LEFT, padx=10)
        
        # Plot button
        plot_btn = ttk.Button(control_frame, text="Generate Histogram", command=self._generate_histogram)
        plot_btn.pack(pady=10)
        
        # Plot display area
        plot_frame = ttk.LabelFrame(tab, text="Plot", padding="5")
        plot_frame.grid(row=1, column=0, sticky="nsew")
        tab.rowconfigure(1, weight=1)
        
        # Matplotlib figure
        self.hist_fig = Figure(figsize=(8, 6), dpi=100)
        self.hist_canvas = FigureCanvasTkAgg(self.hist_fig, master=plot_frame)
        self.hist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Toolbar
        toolbar = NavigationToolbar2Tk(self.hist_canvas, plot_frame)
        toolbar.update()

    def _generate_histogram(self):
        """Generate a histogram using the ecaplots.histogram_plot utility."""
        if not self.db:
            messagebox.showwarning("Warning", "No database loaded")
            return
            
        prop = self.hist_prop_var.get()
        if not prop:
            messagebox.showwarning("Warning", "Please select a property")
            return
            
        # Parse bins parameter
        bins_val = self.hist_bins_var.get().strip()
        if bins_val.lower() != 'auto':
            try:
                bins_val = int(bins_val)
                if bins_val <= 0: raise ValueError
            except ValueError:
                messagebox.showwarning("Warning", "Bins must be 'auto' or a positive integer.")
                return
                
        try:
            self.hist_fig.clear()
            ax = self.hist_fig.add_subplot(111)
            
            # Apply active filters
            if self.search_criteria:
                db_to_use = self.db.extract(self.db.all_keys(), **self.search_criteria)
            else:
                db_to_use = self.db
            
            histogram_plot(db_to_use, prop, bins=bins_val, log_y=self.hist_log_var.get(), size=ax)
            
            self.hist_canvas.draw()
            self._update_status(f"Histogram generated: {prop}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Histogram generation failed: {str(e)}")
            self._update_status("Histogram generation failed")

    def _clear_histogram_tab(self):
        """Clear the histogram tab."""
        self.hist_fig.clear()
        self.hist_canvas.draw()
        self.hist_prop_var.set("")
        self.hist_bins_var.set("auto")
        self.hist_log_var.set(False)
        
    # ==================== INDIVIDUAL PLOT TAB ====================
    
    def _create_individual_plot_tab(self):
        """Create the Individual Plot tab for single simulation plots."""
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="Individual Plot")
        
        # Left panel - Simulation selection
        left_frame = ttk.LabelFrame(tab, text="Select Simulation", padding="5")
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)
        
        # Dynamic Column Controller
        col_frame = ttk.Frame(left_frame)
        col_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(col_frame, text="Add Column:").pack(side=tk.LEFT)
        self.indiv_col_var = tk.StringVar()
        self.indiv_col_combo = ttk.Combobox(col_frame, textvariable=self.indiv_col_var, state="readonly", width=15)
        self.indiv_col_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(col_frame, text="Add", command=self._add_indiv_column).pack(side=tk.LEFT, padx=2)
        ttk.Button(col_frame, text="Reset", command=self._reset_indiv_columns).pack(side=tk.LEFT, padx=2)
        
        # Treeview for simulations
        self.individual_columns = ["doc_id"]
        self.individual_sim_tree = ttk.Treeview(left_frame, columns=self.individual_columns, show="headings", height=15)
        self.individual_sim_tree.heading("doc_id", text="Doc ID")
        self.individual_sim_tree.column("doc_id", width=80)
        
        sim_scroll = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.individual_sim_tree.yview)
        self.individual_sim_tree.configure(yscrollcommand=sim_scroll.set)
        
        self.individual_sim_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sim_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Auto-plot on selection
        self.individual_sim_tree.bind("<<TreeviewSelect>>", lambda e: self._plot_individual_simulation())
        
        # Right panel - Plot controls and display
        right_frame = ttk.Frame(tab)
        right_frame.grid(row=0, column=1, sticky="nsew")
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(0, weight=1)
        
        # Fit selection & UI controls
        fit_frame = ttk.LabelFrame(right_frame, text="Plot Configuration", padding="5")
        fit_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        self.individual_fit_var = tk.StringVar(value="None")
        ttk.Label(fit_frame, text="Apply Fit:").pack(side=tk.LEFT)
        self.individual_fit_combo = ttk.Combobox(fit_frame, textvariable=self.individual_fit_var, state="readonly", width=20)
        self.individual_fit_combo['values'] = ["None", "FurmanNoPhoto", "FurmanNPMC", "FurmanPhoto"]
        self.individual_fit_combo.pack(side=tk.LEFT, padx=5)
        
        self.individual_central_density_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(fit_frame, text="Use Central Density", variable=self.individual_central_density_var).pack(side=tk.LEFT, padx=10)

        ttk.Label(fit_frame, text="Max X:").pack(side=tk.LEFT)
        self.individual_max_x_var = tk.StringVar()
        self.individual_max_x_entry = ttk.Entry(fit_frame, textvariable=self.individual_max_x_var, width=8)
        self.individual_max_x_entry.pack(side=tk.LEFT, padx=2)
        
        # Optional Plot Button
        plot_btn = ttk.Button(fit_frame, text="Replot (Apply Changes)", command=self._plot_individual_simulation)
        plot_btn.pack(side=tk.LEFT, padx=10)
        
        # Plot display
        plot_frame = ttk.LabelFrame(right_frame, text="Simulation Plot", padding="5")
        plot_frame.grid(row=1, column=0, sticky="nsew")
        right_frame.rowconfigure(1, weight=1)
        
        # Matplotlib figure
        self.individual_fig = Figure(figsize=(10, 6), dpi=100)
        self.individual_canvas = FigureCanvasTkAgg(self.individual_fig, master=plot_frame)
        self.individual_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Toolbar
        toolbar = NavigationToolbar2Tk(self.individual_canvas, plot_frame)
        toolbar.update()
        
    def _populate_individual_options(self):
        """Populate the column selection combobox."""
        if not self.db:
            return
        all_keys = self.db.all_keys()
        self.indiv_col_combo['values'] = [k for k in all_keys if k != "path"]
        
    def _add_indiv_column(self):
        new_col = self.indiv_col_var.get()
        if new_col and new_col not in self.individual_columns:
            self.individual_columns.append(new_col)
            self._update_individual_sim_list()

    def _reset_indiv_columns(self):
        self.individual_columns = ["doc_id"]
        self._update_individual_sim_list()

    def _update_individual_sim_list(self):
        """Update the individual simulation list to match the current columns & database filters."""
        for item in self.individual_sim_tree.get_children():
            self.individual_sim_tree.delete(item)
            
        if not self.db:
            return
            
        # Dynamically set headers
        self.individual_sim_tree.configure(columns=self.individual_columns)
        for col in self.individual_columns:
            self.individual_sim_tree.heading(col, text=col)
            self.individual_sim_tree.column(col, width=120 if col != "doc_id" else 60)
            
        # Iterate over results corresponding to the search filter and fill rows
        simulations = self.db.where(**self.search_criteria)
        for sim in simulations:
            values = []
            for col in self.individual_columns:
                if col == "doc_id":
                    values.append(sim.doc_id)
                else:
                    values.append(self._format_value(sim.get(col, "N/A")))
            self.individual_sim_tree.insert("", tk.END, values=values)
            
    def _plot_individual_simulation(self):
        """Plot selected simulation(s). Auto-triggers on tree selection and allows multi-select."""
        selections = self.individual_sim_tree.selection()
        if not selections or not self.db:
            return
            
        self.individual_fig.clear()
        ax = self.individual_fig.add_subplot(111)
        
        fit_model_name = self.individual_fit_var.get()
        plotted_count = 0
        
        for idx, selection in enumerate(selections):
            sim_item = self.individual_sim_tree.item(selection)
            try:
                doc_id_idx = self.individual_columns.index("doc_id")
                doc_id = int(sim_item['values'][doc_id_idx])
            except (ValueError, IndexError):
                continue
                
            model = ECModel(self.db.db, doc_id)
            path_exists = os.path.exists(model.path) and os.path.exists(os.path.join(model.path, "Pyecltest.mat"))
            
            # Dynamically instantiate the required fit object for this simulation
            fits_to_plot = []
            if fit_model_name != "None":
                if fit_model_name == "FurmanNoPhoto":
                    fits_to_plot.append(FurmanNoPhotoFit())
                elif fit_model_name == "FurmanNPMC":
                    fits_to_plot.append(FurmanNPMCFit(self.db)) # DB injected per sim
                elif fit_model_name == "FurmanPhoto":
                    fits_to_plot.append(FurmanPhotoFit())

            if not path_exists and not fits_to_plot:
                continue
            
            # Determine central density flag based on UI toggle and data availability
            cd_param = self.individual_central_density_var.get() if path_exists else None
            
            # Read Max X override, calculate default if empty
            max_x_str = self.individual_max_x_var.get().strip()
            if not max_x_str:
                try:
                    fit_max_x = (model.cutoff / model.bunch_step) * 1.25
                    # Display the default if only one simulation is selected
                    if len(selections) == 1:
                        self.individual_max_x_var.set(f"{fit_max_x:.1f}")
                except Exception:
                    fit_max_x = 300.0
            else:
                try:
                    fit_max_x = float(max_x_str)
                except ValueError:
                    fit_max_x = 300.0
            
            # Guard rails for plotting empty paths
            if not path_exists and fit_max_x <= 0:
                    fit_max_x = 300.0
            
            model_plot(
                model=model,
                fits=fits_to_plot,
                size=ax,
                show_error=True,
                central_density=cd_param,
                fit_maxX=fit_max_x,
                label=str(doc_id)
            )
                        
            plotted_count += 1
                
        if plotted_count > 0:
            self.individual_fig.tight_layout()
            self.individual_canvas.draw()
            self._update_status(f"Plotted {plotted_count} simulation(s)")
        else:
            self._update_status("No valid data or fits to plot.")
            
    def _clear_individual_plot_tab(self):
        """Clear the individual plot tab."""
        self.individual_fig.clear()
        self.individual_canvas.draw()
        for item in self.individual_sim_tree.get_children():
            self.individual_sim_tree.delete(item)
        self.individual_fit_var.set("None")


def main():
    """Main entry point for the application."""
    root = tk.Tk()
    app = ECAApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()