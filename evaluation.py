'''
No@
June 2025
'''
import tkinter as tk
from tkinter import messagebox
import json
import os
import datetime
import csv

class EvaluationPanel(tk.LabelFrame):
    def __init__(self, parent):
        super().__init__(parent, text="Prediction Evaluation", padx=10, pady=10)

        self.scene_name = ""
        self.unsaved_changes = False

        self.region_evaluation = tk.StringVar(value=" ")
        self.boundary_evaluation = tk.StringVar(value=" ")

        for col in range(4):
            self.grid_columnconfigure(col, weight=1)

        tk.Label(self, text="").grid(row=0, column=0, sticky="w")
        tk.Label(self, text="Region:").grid(row=1, column=0, sticky="w")
        tk.Label(self, text="Boundaries:").grid(row=2, column=0, sticky="w")

        # Region Radiobuttons
        # tk.Label(self, text="Accuracy").grid(row=0, column=1, sticky="nsew")
        tk.Radiobutton(self, text="Highly accurate", variable=self.region_evaluation,
                       value="high").grid(row=1, column=1, sticky="nsew")
        tk.Radiobutton(self, text="Sufficient", variable=self.region_evaluation,
                       value="sufficient").grid(row=1, column=2, sticky="nsew")
        tk.Radiobutton(self, text="Not sufficient", variable=self.region_evaluation,
                       value="not sufficient").grid(row=1, column=3, sticky="nsew")

        # Boundary Radiobuttons
        tk.Label(self, text="Operational Capability").grid(row=0, column=1, columnspan=3, sticky="nsew")
        tk.Radiobutton(self, text="Highly accurate", variable=self.boundary_evaluation,
                       value="high").grid(row=2, column=1, sticky="nsew")
        tk.Radiobutton(self, text="Sufficient", variable=self.boundary_evaluation,
                       value="sufficient").grid(row=2, column=2, sticky="nsew")
        tk.Radiobutton(self, text="Not sufficient", variable=self.boundary_evaluation,
                       value="not sufficient").grid(row=2, column=3, sticky="nsew")

        tk.Label(self, text="Notes:").grid(row=0, column=4, sticky="w", padx = 10)
        self.notes_text = tk.Text(self, width=40, height=6)
        self.notes_text.grid(row=1, column=4, rowspan=3, pady=(10, 0), padx = 10)

        self.save_button = tk.Button(self, text="Save Evaluation", command=self.save_evaluation)
        self.save_button.grid(row=0, column=0, pady=10)

        # Add traces to mark unsaved changes
        self._trace_region_id = self.region_evaluation.trace_add("write", lambda *args: self._mark_unsaved())
        self._trace_boundary_id = self.boundary_evaluation.trace_add("write", lambda *args: self._mark_unsaved())
        self.notes_text.bind("<KeyRelease>", lambda e: self._mark_unsaved())

    def _mark_unsaved(self):
        if not self.unsaved_changes:
            self.unsaved_changes = True
            self.config(text="* Prediction Evaluation")

    def set_scene_name(self, name):
        self.scene_name = name
        self.after(100, self.load_existing_evaluation)

    def load_existing_evaluation(self):
        filename = os.path.join("evaluations", "evaluation_data.json")
        if os.path.exists(filename):
            with open(filename, "r") as f:
                try:
                    data = json.load(f)
                    if self.scene_name in data:
                        scene_data = data[self.scene_name]
                        self.region_evaluation.set(scene_data.get("region", "").strip())
                        self.boundary_evaluation.set(scene_data.get("boundaries", "").strip())

                        self.notes_text.config(state=tk.NORMAL)
                        self.notes_text.delete("1.0", tk.END)
                        self.notes_text.insert(tk.END, scene_data.get("notes", ""))
                        
                        self.notes_text.focus_set()
                        # self.notes_text.mark_set("insert", "1.0")
                       
                        self.unsaved_changes = False
                        self.config(text="Prediction Evaluation")
                        messagebox.showinfo("Loaded", f"Existing evaluation loaded for scene '{self.scene_name}'")
                    else:
                        self.reset_fields()  # reset if no data for this scene
                except json.JSONDecodeError:
                    pass
        else:
            self.reset_fields()

    def save_evaluation(self):
        if not self.scene_name:
            messagebox.showerror("Error", "Scene name is not set.")
            return False

        region_accuracy = self.region_evaluation.get().strip()
        boundary_accuracy = self.boundary_evaluation.get().strip()
        notes = self.notes_text.get("1.0", tk.END).strip()

        if region_accuracy == '' and boundary_accuracy == '':
            messagebox.showerror("Warning", "Evaluation incomplete")
            return False
        if region_accuracy == '':
            messagebox.showerror("Warning", "Evaluation incomplete for regions")
            return False
        if boundary_accuracy == '':
            messagebox.showerror("Warning", "Evaluation incomplete for boundaries")
            return False

        new_data = {
            self.scene_name: {
                "region": region_accuracy,
                "boundaries": boundary_accuracy,
                "notes": notes
            }
        }

        os.makedirs("evaluations", exist_ok=True)
        filename = os.path.join("evaluations", "evaluation_data.json")

        if os.path.exists(filename):
            with open(filename, "r") as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = {}
        else:
            existing_data = {}

        overwrite = False
        if self.scene_name in existing_data:
            overwrite = messagebox.askyesno("Overwrite?",
                f"An entry already exists for scene '{self.scene_name}'. Do you want to replace it?")
            if not overwrite:
                return False

        existing_data.update(new_data)
        with open(filename, "w") as f:
            json.dump(existing_data, f, indent=4)

        self._save_to_csv(self.scene_name, region_accuracy, boundary_accuracy, notes)

        self.unsaved_changes = False
        self.config(text="Prediction Evaluation")
        self._show_silent_popup("Saved" if not overwrite else "Updated", f"Evaluation saved to {filename}")

        return True

    def _save_to_csv(self, scene_name, region_eval, boundary_eval, notes):
        csv_file = os.path.join("evaluations", "evaluation_data.csv")
        file_exists = os.path.isfile(csv_file)
        fieldnames = ["scene_name", "region", "boundaries", "notes", "timestamp"]

        rows = []
        if file_exists:
            with open(csv_file, "r", newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

        rows = [row for row in rows if row["scene_name"] != scene_name]

        new_row = {
            "scene_name": scene_name,
            "region": region_eval,
            "boundaries": boundary_eval,
            "notes": notes,
            "timestamp": datetime.datetime.now().isoformat()
        }
        rows.append(new_row)

        with open(csv_file, "w", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _show_silent_popup(self, title, message):
        popup = tk.Toplevel(self)
        popup.title(title)
        popup.resizable(False, False)
        popup.transient(self)
        popup.grab_set()

        tk.Label(popup, text=message, padx=20, pady=10).pack()
        tk.Button(popup, text="OK", command=popup.destroy).pack(pady=(0, 10))

        popup.update_idletasks()
        w = popup.winfo_width()
        h = popup.winfo_height()
        x = self.winfo_rootx() + (self.winfo_width() - w) // 2
        y = self.winfo_rooty() + (self.winfo_height() - h) // 2
        popup.geometry(f"{w}x{h}+{x}+{y}")

        # Wait until popup is closed
        popup.wait_window()

    def set_enabled(self, enabled=True):
        # bg_color = "SystemButtonFace" if enabled else "#dddddd"
        state = tk.NORMAL if enabled else tk.DISABLED

        # Change state and background color for all interactive widgets
        for child in self.winfo_children():
            widget_type = child.winfo_class()
            if widget_type in ("Button", "Radiobutton", "Text"):
                try:
                    child.configure(state=state)
                except:
                    pass
            # try:
            #     child.configure(bg=bg_color)
            # except:
            #     pass

    def reset_fields(self):
        # Temporarily remove traces
        self.region_evaluation.trace_remove("write", self._trace_region_id)
        self.boundary_evaluation.trace_remove("write", self._trace_boundary_id)
    
        # Reset values without triggering _mark_unsaved
        self.region_evaluation.set(" ")
        self.boundary_evaluation.set(" ")

        # Reattach traces
        self._trace_region_id = self.region_evaluation.trace_add("write", lambda *args: self._mark_unsaved())
        self._trace_boundary_id = self.boundary_evaluation.trace_add("write", lambda *args: self._mark_unsaved())

        # Reset notes field
        self.notes_text.config(state=tk.NORMAL)
        self.notes_text.delete("1.0", tk.END)

        # Clear unsaved state
        self.unsaved_changes = False
        self.config(text="Prediction Evaluation")

if __name__ == '__main__':
    root = tk.Tk()
    root.title("Evaluation Panel")

    panel = EvaluationPanel(root)
    panel.pack(padx=10, pady=10)

    panel.set_scene_name("example_scene")

    root.mainloop()
