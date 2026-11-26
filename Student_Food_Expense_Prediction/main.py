"""Student Daily Food Expense Prediction System - main Tkinter application."""
import tkinter as tk
from tkinter import ttk

APP_TITLE = "Student Food Expense Prediction System"

# A clean academic theme with mathematical accents.
BG = "#F3F6FC"
NAVY = "#17243A"
NAVY_LIGHT = "#263954"
BLUE = "#4169E1"
TEAL = "#20B8A5"
TEXT = "#1E293B"
MUTED = "#718096"
WHITE = "#FFFFFF"
BORDER = "#E2E8F0"


class FoodExpenseApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1160x720")
        self.minsize(960, 600)
        self.configure(bg=BG)

        self.pages = [
            ("Dashboard", "⌂"),
            ("Dataset Management", "▦"),
            ("Data Exploration", "◫"),
            ("Linear Algebra", "∑"),
            ("Eigenvalues & PCA", "λ"),
            ("Regression Analysis", "ƒ"),
            ("Prediction", "↗"),
        ]
        self.buttons = {}
        self._build_shell()
        self.show_page("Dashboard")

    def _build_shell(self):
        self.sidebar = tk.Frame(self, bg=NAVY, width=238)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        brand = tk.Frame(self.sidebar, bg=NAVY)
        brand.pack(fill="x", padx=20, pady=(25, 30))
        tk.Label(
            brand, text="∑", font=("Cambria Math", 28, "bold"),
            fg=WHITE, bg=BLUE, width=2, pady=3
        ).pack(side="left", padx=(0, 10))
        brand_words = tk.Frame(brand, bg=NAVY)
        brand_words.pack(side="left")
        tk.Label(
            brand_words, text="FOOD EXPENSE", fg=WHITE, bg=NAVY,
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w")
        tk.Label(
            brand_words, text="PREDICTION SYSTEM", fg="#AAB8CC", bg=NAVY,
            font=("Segoe UI", 8)
        ).pack(anchor="w", pady=(3, 0))

        tk.Label(
            self.sidebar, text="WORKSPACE", fg="#8FA1BA", bg=NAVY,
            font=("Segoe UI", 8, "bold")
        ).pack(anchor="w", padx=22, pady=(0, 10))

        for name, symbol in self.pages:
            btn = tk.Button(
                self.sidebar, text=f"  {symbol}     {name}", anchor="w",
                font=("Segoe UI", 10), fg="#DCE5F2", bg=NAVY,
                activeforeground=WHITE, activebackground=NAVY_LIGHT,
                relief="flat", bd=0, padx=15, pady=12, cursor="hand2",
                command=lambda page=name: self.show_page(page)
            )
            btn.pack(fill="x", padx=11, pady=2)
            self.buttons[name] = btn

        tk.Frame(self.sidebar, bg="#34445D", height=1).pack(
            side="bottom", fill="x", padx=18, pady=(10, 12)
        )
        tk.Label(
            self.sidebar, text="LINEAR ALGEBRA  •  PCA  •  REGRESSION",
            fg="#C5D0DF", bg=NAVY, font=("Segoe UI", 7, "bold")
        ).pack(side="bottom", anchor="w", padx=18, pady=(0, 18))

        self.main_area = tk.Frame(self, bg=BG)
        self.main_area.pack(side="right", fill="both", expand=True)

        top = tk.Frame(
            self.main_area, bg=WHITE, height=64,
            highlightbackground=BORDER, highlightthickness=1
        )
        top.pack(fill="x")
        top.pack_propagate(False)
        self.heading = tk.Label(
            top, text="Dashboard", font=("Segoe UI", 16, "bold"),
            fg=TEXT, bg=WHITE
        )
        self.heading.pack(side="left", padx=28)
        tk.Label(
            top, text="STUDENT ANALYTICS", font=("Segoe UI", 8, "bold"),
            fg=BLUE, bg="#EDF2FF", padx=12, pady=7
        ).pack(side="right", padx=25)

        self.body = tk.Frame(self.main_area, bg=BG)
        self.body.pack(fill="both", expand=True, padx=28, pady=22)

    def _clear_body(self):
        for child in self.body.winfo_children():
            child.destroy()

    def _heading(self, title, subtitle):
        tk.Label(
            self.body, text=title, font=("Segoe UI", 21, "bold"),
            fg=TEXT, bg=BG
        ).pack(anchor="w", pady=(5, 4))
        tk.Label(
            self.body, text=subtitle, font=("Segoe UI", 10),
            fg=MUTED, bg=BG
        ).pack(anchor="w", pady=(0, 20))

    def _card(self, parent, title, value, detail, accent):
        card = tk.Frame(
            parent, bg=WHITE, highlightbackground=BORDER, highlightthickness=1
        )
        tk.Frame(card, bg=accent, height=4).pack(fill="x")
        inner = tk.Frame(card, bg=WHITE)
        inner.pack(fill="both", expand=True, padx=17, pady=15)
        tk.Label(
            inner, text=title.upper(), font=("Segoe UI", 8, "bold"),
            fg=MUTED, bg=WHITE
        ).pack(anchor="w")
        tk.Label(
            inner, text=value, font=("Segoe UI", 21, "bold"),
            fg=TEXT, bg=WHITE
        ).pack(anchor="w", pady=(7, 3))
        tk.Label(
            inner, text=detail, font=("Segoe UI", 8),
            fg=MUTED, bg=WHITE
        ).pack(anchor="w")
        return card

    def show_page(self, page):
        self.heading.config(text=page)
        for name, button in self.buttons.items():
            button.config(
                bg=BLUE if name == page else NAVY,
                fg=WHITE if name == page else "#DCE5F2"
            )
        self._clear_body()

        if page == "Dashboard":
            self._dashboard()
        elif page == "Dataset Management":
            self._dataset_page()
        else:
            self._placeholder(page)

    def _dashboard(self):
        self._heading(
            "Welcome to your analysis workspace",
            "Explore student food expenses through data analysis and mathematical modeling."
        )

        cards = tk.Frame(self.body, bg=BG)
        cards.pack(fill="x")
        for i, item in enumerate([
            ("INPUT VARIABLES", "4", "Allowance and spending habits", BLUE),
            ("PREDICTION TARGET", "₱ PHP", "Daily food expense", TEAL),
            ("MODEL APPROACH", "PCA + LR", "Compare regression models", "#8B5CF6"),
        ]):
            card = self._card(cards, *item)
            card.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 10, 0))
            cards.grid_columnconfigure(i, weight=1)

        lower = tk.Frame(self.body, bg=BG)
        lower.pack(fill="both", expand=True, pady=(20, 0))

        overview = tk.Frame(
            lower, bg=WHITE, highlightbackground=BORDER, highlightthickness=1
        )
        overview.pack(side="left", fill="both", expand=True, padx=(0, 10))
        tk.Label(
            overview, text="PROJECT OVERVIEW", font=("Segoe UI", 9, "bold"),
            fg=BLUE, bg=WHITE
        ).pack(anchor="w", padx=20, pady=(19, 10))
        tk.Label(
            overview,
            text=(
                "This system examines how students' allowance and spending habits "
                "relate to their daily food expenses. It applies linear algebra, "
                "Principal Component Analysis (PCA), and Linear Regression to "
                "analyze the dataset and compare prediction performance."
            ),
            font=("Segoe UI", 10), fg=TEXT, bg=WHITE,
            wraplength=490, justify="left"
        ).pack(anchor="w", padx=20, pady=(0, 16))
        formula = tk.Frame(overview, bg="#EEF3FF")
        formula.pack(fill="x", padx=20, pady=(0, 20))
        tk.Label(
            formula, text="ŷ = β₀ + β₁x₁ + β₂x₂ + ⋯ + βₙxₙ",
            font=("Cambria Math", 18), fg=BLUE, bg="#EEF3FF", pady=15
        ).pack()

        quick = tk.Frame(
            lower, bg=WHITE, width=245,
            highlightbackground=BORDER, highlightthickness=1
        )
        quick.pack(side="right", fill="y")
        quick.pack_propagate(False)
        tk.Label(
            quick, text="QUICK NAVIGATION", font=("Segoe UI", 9, "bold"),
            fg=BLUE, bg=WHITE
        ).pack(anchor="w", padx=17, pady=(19, 12))
        for label, destination in [
            ("Open Dataset Management", "Dataset Management"),
            ("Explore Linear Algebra", "Linear Algebra"),
            ("View Eigenvalues & PCA", "Eigenvalues & PCA"),
            ("Open Prediction", "Prediction"),
        ]:
            tk.Button(
                quick, text=label, anchor="w", font=("Segoe UI", 9),
                fg=TEXT, bg="#F4F7FC", activebackground="#E5ECFA",
                relief="flat", bd=0, padx=11, pady=10, cursor="hand2",
                command=lambda target=destination: self.show_page(target)
            ).pack(fill="x", padx=13, pady=4)

        tk.Label(
            self.body,
            text="Analysis results and model metrics will appear after the project modules are integrated.",
            font=("Segoe UI", 8), fg=MUTED, bg=BG
        ).pack(anchor="w", pady=(12, 0))

    def _dataset_page(self):
        self._heading(
            "Dataset Management",
            "Prepare, load, and validate the student survey dataset."
        )
        panel = tk.Frame(
            self.body, bg=WHITE, highlightbackground=BORDER, highlightthickness=1
        )
        panel.pack(fill="x", pady=(0, 18))
        tk.Label(
            panel, text="EXPECTED DATA FILE", font=("Segoe UI", 8, "bold"),
            fg=MUTED, bg=WHITE
        ).pack(anchor="w", padx=20, pady=(17, 6))
        tk.Label(
            panel, text="data/Student_Daily_Food_Expenses_Model_Ready.csv",
            font=("Consolas", 10), fg=BLUE, bg=WHITE
        ).pack(anchor="w", padx=20, pady=(0, 17))

        tk.Label(
            self.body, text="Dataset variables", font=("Segoe UI", 12, "bold"),
            fg=TEXT, bg=BG
        ).pack(anchor="w", pady=(0, 9))
        table = ttk.Treeview(
            self.body, columns=("variable", "meaning", "role"),
            show="headings", height=5
        )
        for col, title, width in [
            ("variable", "VARIABLE", 235),
            ("meaning", "DESCRIPTION", 450),
            ("role", "ROLE", 110),
        ]:
            table.heading(col, text=title)
            table.column(col, width=width, anchor="w")
        for row in [
            ("DailyAllowance_PHP", "Student's daily allowance in pesos", "Input"),
            ("MealsPurchased", "Number of meals purchased", "Input"),
            ("SnacksPurchased", "Number of snacks purchased", "Input"),
            ("CampusHours", "Hours spent on campus", "Input"),
            ("DailyFoodExpense_PHP", "Daily food expense in pesos", "Target"),
        ]:
            table.insert("", "end", values=row)
        table.pack(fill="x")
        tk.Label(
            self.body,
            text="Dataset loading and preview will be connected to modules/dataset_manager.py.",
            font=("Segoe UI", 8), fg=MUTED, bg=BG
        ).pack(anchor="w", pady=12)

    def _placeholder(self, page):
        details = {
            "Data Exploration": "Descriptive statistics, distributions, and dataset summaries.",
            "Linear Algebra": "Matrix dimensions, operations, rank, vector spaces, determinants, and conditioning.",
            "Eigenvalues & PCA": "Covariance matrix, eigenvalues, eigenvectors, explained variance, and principal components.",
            "Regression Analysis": "Train and compare the full-feature and PCA-based regression models.",
            "Prediction": "Enter a student's information and estimate their daily food expense.",
        }
        self._heading(page, "Mathematical analysis workspace")
        panel = tk.Frame(
            self.body, bg=WHITE, highlightbackground=BORDER, highlightthickness=1
        )
        panel.pack(fill="x", pady=5)
        tk.Label(
            panel, text=details.get(page, "Project section"),
            font=("Segoe UI", 11), fg=TEXT, bg=WHITE,
            wraplength=760, justify="left"
        ).pack(anchor="w", padx=22, pady=(22, 12))
        tk.Label(
            panel,
            text="Integration status: This page is ready for its assigned module.",
            font=("Segoe UI", 9), fg=MUTED, bg=WHITE
        ).pack(anchor="w", padx=22, pady=(0, 22))


if __name__ == "__main__":
    app = FoodExpenseApp()
    app.mainloop()
