import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import scipy.sparse as sp 
from scipy.sparse.linalg import spsolve
import copy 
import os
import threading
import time

class SimulateurBronchique(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Simulateur du Système Bronchique - Écoulement dans les Tubes Souples")
        self.geometry("1400x900")
        self.configure(bg='#f0f0f0')
        
        # Variables pour stocker les résultats
        self.resultats = {}
        self.figures_actuelles = []
        
        self.creer_interface()
        
    def creer_interface(self):
        # Style moderne
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook.Tab', font=('Arial', 10, 'bold'), padding=[10, 5])
        
        # Frame principale
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Notebook (onglets)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Création des onglets
        self.creer_onglet_introduction()
        self.creer_onglet_edp_simple()
        self.creer_onglet_laplacien()
        self.creer_onglet_tube_rigide()
        self.creer_onglet_geometrie_variable()
        self.creer_onglet_tube_souple()
        self.creer_onglet_toutes_generations()
        
        # Barre de statut
        self.status_var = tk.StringVar(value="Prêt")
        status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def creer_onglet_introduction(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Introduction")
        
        # Titre
        titre = tk.Label(frame, text="SIMULATEUR DU SYSTÈME BRONCHIQUE", 
                        font=('Arial', 16, 'bold'), fg='#2c3e50')
        titre.pack(pady=20)
        
        # Description
        desc_text = """
Ce simulateur permet d'étudier l'écoulement d'un fluide dans les bronches 
en tenant compte de la déformation élastique des parois.

FONCTIONNALITÉS PRINCIPALES :
• Résolution d'EDP simples (bases mathématiques)
• Simulation d'écoulement en tube rigide
• Géométries variables (cylindrique et aléatoire)
• Tube souple avec interaction fluide-structure
• Simulation complète de l'arbre bronchique (17 générations)
• Analyse du paradoxe de l'expiration forcée

UTILISATION :
1. Naviguez entre les onglets pour différentes simulations
2. Ajustez les paramètres dans le panneau de contrôle
3. Lancez la simulation avec le bouton dédié
4. Visualisez les résultats dans la zone graphique

AUTEUR :
Achraf Zekri

DATE : 11/11/2025
        """
        
        desc = tk.Label(frame, text=desc_text, font=('Arial', 11), 
                       justify=tk.LEFT, bg='#f8f9fa', relief=tk.RAISED, padx=20, pady=20)
        desc.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)
        
    def creer_onglet_edp_simple(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="EDP Simple")
        
        # Panneau de contrôle
        control_frame = ttk.LabelFrame(frame, text="Paramètres EDP Simple", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Paramètres
        ttk.Label(control_frame, text="Longueur L:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.edp_L = tk.DoubleVar(value=10.0)
        ttk.Entry(control_frame, textvariable=self.edp_L, width=10).grid(row=0, column=1, pady=2)
        
        ttk.Label(control_frame, text="Points N:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.edp_N = tk.IntVar(value=100)
        ttk.Entry(control_frame, textvariable=self.edp_N, width=10).grid(row=1, column=1, pady=2)
        
        ttk.Label(control_frame, text="Constante C:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.edp_C = tk.DoubleVar(value=10.0)
        ttk.Entry(control_frame, textvariable=self.edp_C, width=10).grid(row=2, column=1, pady=2)
        
        ttk.Label(control_frame, text="Condition F1:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.edp_F1 = tk.DoubleVar(value=0.0)
        ttk.Entry(control_frame, textvariable=self.edp_F1, width=10).grid(row=3, column=1, pady=2)
        
        ttk.Label(control_frame, text="Condition FN:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.edp_FN = tk.DoubleVar(value=0.0)
        ttk.Entry(control_frame, textvariable=self.edp_FN, width=10).grid(row=4, column=1, pady=2)
        
        # Bouton simulation
        ttk.Button(control_frame, text="Lancer Simulation", 
                  command=self.lancer_edp_simple).grid(row=5, column=0, columnspan=2, pady=10)
        
        # Zone graphique
        self.edp_figure_frame = ttk.Frame(frame)
        self.edp_figure_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def creer_onglet_laplacien(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Laplacien 2D")
        
        # Panneau de contrôle
        control_frame = ttk.LabelFrame(frame, text="Paramètres Laplacien", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Paramètres
        params = [
            ("Points n:", "lapl_n", 40),
            ("Points m:", "lapl_m", 80),
            ("Longueur L:", "lapl_L", 5.32e-2),
            ("Largeur l:", "lapl_l", 7.54e-2),
            ("Constante C:", "lapl_C", -10.0)
        ]
        
        for i, (label, var_name, default) in enumerate(params):
            ttk.Label(control_frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=2)
            var = tk.DoubleVar(value=default)
            setattr(self, var_name, var)
            ttk.Entry(control_frame, textvariable=var, width=10).grid(row=i, column=1, pady=2)
        
        # Bouton simulation
        ttk.Button(control_frame, text="Lancer Simulation", 
                  command=self.lancer_laplacien).grid(row=len(params), column=0, columnspan=2, pady=10)
        
        # Zone graphique
        self.lapl_figure_frame = ttk.Frame(frame)
        self.lapl_figure_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def creer_onglet_tube_rigide(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Tube Rigide")
        
        # Panneau de contrôle
        control_frame = ttk.LabelFrame(frame, text="Paramètres Tube Rigide", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Paramètres
        params = [
            ("Points n:", "rigide_n", 100),
            ("Points m:", "rigide_m", 200),
            ("Longueur L:", "rigide_L", 5.32e-2),
            ("Largeur l:", "rigide_l", 7.54e-2),
            ("Viscosité C:", "rigide_C", 1.8),
            ("Pression Entrée:", "rigide_P1", 400),
            ("Pression Sortie:", "rigide_P2", 200)
        ]
        
        for i, (label, var_name, default) in enumerate(params):
            ttk.Label(control_frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=2)
            var = tk.DoubleVar(value=default)
            setattr(self, var_name, var)
            ttk.Entry(control_frame, textvariable=var, width=12).grid(row=i, column=1, pady=2)
        
        # Bouton simulation
        ttk.Button(control_frame, text="Lancer Simulation", 
                  command=self.lancer_tube_rigide).grid(row=len(params), column=0, columnspan=2, pady=10)
        
        # Zone graphique
        self.rigide_figure_frame = ttk.Frame(frame)
        self.rigide_figure_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def creer_onglet_geometrie_variable(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Géométrie Variable")
        
        # Panneau de contrôle
        control_frame = ttk.LabelFrame(frame, text="Paramètres Géométrie", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Type de géométrie
        ttk.Label(control_frame, text="Type de géométrie:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.geom_type = tk.StringVar(value="cylindrique")
        ttk.Radiobutton(control_frame, text="Cylindrique", variable=self.geom_type, 
                       value="cylindrique").grid(row=1, column=0, sticky=tk.W)
        ttk.Radiobutton(control_frame, text="Aléatoire", variable=self.geom_type, 
                       value="aleatoire").grid(row=2, column=0, sticky=tk.W)
        
        # Paramètres communs
        common_params = [
            ("Points n:", "geom_n", 100),
            ("Points m:", "geom_m", 250),
            ("Longueur L:", "geom_L", 5.32e-2),
            ("Largeur l:", "geom_l", 7.54e-2),
            ("Viscosité C:", "geom_C", 1.8),
            ("Pression Entrée:", "geom_P1", 400),
            ("Pression Sortie:", "geom_P2", 200)
        ]
        
        for i, (label, var_name, default) in enumerate(common_params):
            ttk.Label(control_frame, text=label).grid(row=i+3, column=0, sticky=tk.W, pady=2)
            var = tk.DoubleVar(value=default)
            setattr(self, var_name, var)
            ttk.Entry(control_frame, textvariable=var, width=12).grid(row=i+3, column=1, pady=2)
        
        # Bouton simulation
        ttk.Button(control_frame, text="Lancer Simulation", 
                  command=self.lancer_geometrie_variable).grid(row=len(common_params)+3, column=0, columnspan=2, pady=10)
        
        # Zone graphique
        self.geom_figure_frame = ttk.Frame(frame)
        self.geom_figure_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def creer_onglet_tube_souple(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Tube Souple")
        
        # Panneau de contrôle
        control_frame = ttk.LabelFrame(frame, text="Paramètres Tube Souple", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Paramètres physiques du tube
        params_physiques = [
            ("Rayon maximal R0 (m):", "souple_R0", 0.00068),
            ("Paramètre forme α0:", "souple_alpha0", 0.102),
            ("Exposant compression n1:", "souple_n1", 1.0),
            ("Exposant expansion n2:", "souple_n2", 10.0),
            ("Pression compression P1:", "souple_P1", 53.0),
            ("Pression expansion P2:", "souple_P2", -4635.0),
            ("Pression externe P0:", "souple_P0", 0.0)
        ]
        
        for i, (label, var_name, default) in enumerate(params_physiques):
            ttk.Label(control_frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=2)
            var = tk.DoubleVar(value=default)
            setattr(self, var_name, var)
            ttk.Entry(control_frame, textvariable=var, width=12).grid(row=i, column=1, pady=2)
        
        # Paramètres de simulation
        params_simulation = [
            ("Longueur L (m):", "souple_L", 0.0047),
            ("Pression entrée (Pa):", "souple_P_entree", 490.5),
            ("Pression sortie (Pa):", "souple_P_sortie", 451.3),
            ("Points n:", "souple_n", 100),
            ("Points m:", "souple_m", 100),
            ("Facteur zoom:", "souple_zoom", 1),
            ("Viscosité C:", "souple_C", 1.8e-5),
            ("Itérations max:", "souple_iter_max", 1000),
            ("Tolérance:", "souple_tolerance", 1e-5)
        ]
        
        for i, (label, var_name, default) in enumerate(params_simulation):
            ttk.Label(control_frame, text=label).grid(row=i+len(params_physiques), column=0, sticky=tk.W, pady=2)
            var = tk.DoubleVar(value=default)
            setattr(self, var_name, var)
            ttk.Entry(control_frame, textvariable=var, width=12).grid(row=i+len(params_physiques), column=1, pady=2)
        
        # Bouton simulation
        ttk.Button(control_frame, text="Lancer Simulation Tube Souple", 
                  command=self.lancer_tube_souple).grid(row=len(params_physiques)+len(params_simulation), 
                                                      column=0, columnspan=2, pady=10)
        
        # Zone graphique
        self.souple_figure_frame = ttk.Frame(frame)
        self.souple_figure_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def creer_onglet_toutes_generations(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Toutes Générations")
        
        # Panneau de contrôle
        control_frame = ttk.LabelFrame(frame, text="Paramètres Toutes Générations", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Paramètres de simulation
        params = [
            ("Type de simulation:", "gen_type", "expiration_forcee"),
            ("Itérations max par génération:", "gen_iter_max", 100),
            ("Tolérance convergence:", "gen_tolerance", 1e-5),
            ("Viscosité C:", "gen_C", 1.8e-5)
        ]
        
        # Type de simulation
        ttk.Label(control_frame, text="Type de simulation:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.gen_type = tk.StringVar(value="expiration_forcee")
        ttk.Radiobutton(control_frame, text="Expiration forcée", variable=self.gen_type, 
                       value="expiration_forcee").grid(row=1, column=0, sticky=tk.W)
        ttk.Radiobutton(control_frame, text="Inspiration", variable=self.gen_type, 
                       value="inspiration").grid(row=2, column=0, sticky=tk.W)
        ttk.Radiobutton(control_frame, text="Repos", variable=self.gen_type, 
                       value="repos").grid(row=3, column=0, sticky=tk.W)
        
        # Autres paramètres
        for i, (label, var_name, default) in enumerate(params[1:], 4):
            ttk.Label(control_frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=2)
            var = tk.DoubleVar(value=default)
            setattr(self, var_name, var)
            ttk.Entry(control_frame, textvariable=var, width=12).grid(row=i, column=1, pady=2)
        
        # Bouton simulation
        ttk.Button(control_frame, text="Lancer Simulation Toutes Générations", 
                  command=self.lancer_toutes_generations).grid(row=8, column=0, columnspan=2, pady=10)
        
        # Barre de progression
        ttk.Label(control_frame, text="Progression:").grid(row=9, column=0, sticky=tk.W, pady=5)
        self.gen_progress = ttk.Progressbar(control_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.gen_progress.grid(row=9, column=1, pady=5, sticky=tk.W+tk.E)
        
        self.gen_progress_label = ttk.Label(control_frame, text="0/17 générations")
        self.gen_progress_label.grid(row=10, column=0, columnspan=2, pady=2)
        
        # Zone graphique
        self.gen_figure_frame = ttk.Frame(frame)
        self.gen_figure_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    # Méthodes pour lancer les simulations
    def lancer_edp_simple(self):
        self.status_var.set("Simulation EDP simple en cours...")
        
        # Récupération des paramètres
        L = self.edp_L.get()
        N = int(self.edp_N.get())
        C = self.edp_C.get()
        F1 = self.edp_F1.get()
        FN = self.edp_FN.get()
        
        # Exécution dans un thread séparé
        thread = threading.Thread(target=self.executer_edp_simple, 
                                args=(L, N, C, F1, FN))
        thread.daemon = True
        thread.start()
        
    def executer_edp_simple(self, L, N, C, F1, FN):
        try:
            # Code 1 exact - sans modification
            hx = L/N
            
            def matrice(N, C, hx, F1, FN):
                A, B = np.zeros((N, N)), np.zeros(N)
                
                # Conditions aux limites
                A[0, 0] = 1.0
                B[0] = F1
                A[N-1, N-1] = 1.0
                B[N-1] = FN
                
                # Points intérieurs (schéma différences finies)
                for i in range(1, N - 1):
                    A[i, i-1] = 1/hx**2
                    A[i, i]   = -2/hx**2
                    A[i, i+1] = 1/hx**2
                    B[i] = C
                    
                return A, B

            # Résolution
            A, B = matrice(N, C, hx, F1, FN)
            S = np.linalg.solve(A, B)
            
            # Création du graphique
            self.after(0, self.afficher_graphique_edp, S, L, N, C)
            
        except Exception as e:
            self.after(0, self.afficher_erreur, f"Erreur EDP simple: {str(e)}")
            
    def afficher_graphique_edp(self, S, L, N, C):
        # Nettoyer la frame
        for widget in self.edp_figure_frame.winfo_children():
            widget.destroy()
            
        # Créer la figure
        fig = Figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        
        X = np.linspace(0, L, N)
        ax.plot(X, S, 'b-', linewidth=2)
        ax.set_title(f"Solution de $f''(x) = {C}$", fontsize=12)
        ax.set_xlabel("Position x (m)")
        ax.set_ylabel("Valeur de f(x)")
        ax.grid(True, alpha=0.3)
        
        # Intégrer dans Tkinter
        canvas = FigureCanvasTkAgg(fig, self.edp_figure_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.status_var.set("Simulation EDP simple terminée")
        
    def lancer_laplacien(self):
        self.status_var.set("Simulation Laplacien en cours...")
        
        # Récupération des paramètres
        n = int(self.lapl_n.get())
        m = int(self.lapl_m.get())
        L = self.lapl_L.get()
        l_val = self.lapl_l.get()
        C = self.lapl_C.get()
        
        thread = threading.Thread(target=self.executer_laplacien, 
                                args=(n, m, L, l_val, C))
        thread.daemon = True
        thread.start()
        
    def executer_laplacien(self, n, m, L, l_val, C):
        try:
            # Code 2 exact - sans modification
            hx, hy = L/n, l_val/m 
            N = n * m

            def be(i, j, n): return j * n + i
            def re(k, n): return (k % n, k // n)

            def matrixes(n, m, C, hx, hy):
                N = m * n
                A, B = np.zeros((N, N)), np.zeros(N)
                
                for K in range(N):
                    i, j = re(K, n)

                    if j == 0 or j == m - 1:
                        A[K, K] = 1.0
                        B[K] = 0.0
                    elif i == 0:
                        A[K, K] = 1.0
                        A[K, be(i + 1, j, n)] = -1.0
                    elif i == n - 1:
                        A[K, K] = 1.0
                        A[K, be(i - 1, j, n)] = -1.0
                    else:
                        A[K, K] = -2 * (1/hx**2 + 1/hy**2)
                        A[K, be(i - 1, j, n)] = 1/hx**2
                        A[K, be(i + 1, j, n)] = 1/hx**2
                        A[K, be(i, j - 1, n)] = 1/hy**2
                        A[K, be(i, j + 1, n)] = 1/hy**2
                        B[K] = C
                        
                return A, B    

            A, B = matrixes(n, m, C, hx, hy)
            S = np.linalg.solve(A, B)
            
            self.after(0, self.afficher_graphique_laplacien, S, n, m, L, l_val, C)
            
        except Exception as e:
            self.after(0, self.afficher_erreur, f"Erreur Laplacien: {str(e)}")
            
    def afficher_graphique_laplacien(self, S, n, m, L, l_val, C):
        for widget in self.lapl_figure_frame.winfo_children():
            widget.destroy()
            
        fig = Figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        
        im = ax.imshow(S.reshape((m, n)), extent=[0, L, 0, l_val], 
                      origin='lower', aspect='auto', cmap='viridis')
        fig.colorbar(im, ax=ax, label='Valeur de S')
        ax.set_xlabel('Position x (m)')
        ax.set_ylabel('Position y (m)')
        ax.set_title(f'Solution de $\\Delta U = {C}$', fontsize=12)
        
        canvas = FigureCanvasTkAgg(fig, self.lapl_figure_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.status_var.set("Simulation Laplacien terminée")
        
    def lancer_tube_rigide(self):
        self.status_var.set("Simulation Tube Rigide en cours...")
        
        n = int(self.rigide_n.get())
        m = int(self.rigide_m.get())
        L = self.rigide_L.get()
        l_val = self.rigide_l.get()
        C = self.rigide_C.get()
        P1 = self.rigide_P1.get()
        P2 = self.rigide_P2.get()
        
        thread = threading.Thread(target=self.executer_tube_rigide, 
                                args=(n, m, L, l_val, C, P1, P2))
        thread.daemon = True
        thread.start()
        
    def executer_tube_rigide(self, n, m, L, l_val, C, P1, P2):
        try:
            # Code 3 exact - sans modification
            hx, hy = L/n, l_val/m 
            
            Np, Nu, Nv = n*m, (n+1)*m, n*(m+1)
            N = Np + Nu + Nv

            def idx_P(i, j, n): return j * n + i 
            def idx_U(i, j, n): return j * (n + 1) + i + Np
            def idx_V(i, j, n, m): return j * n + i + Np + Nu

            def matrixes(n, m, C, hx, hy, P1, P2):
                Np, Nu, Nv = n*m, (n+1)*m, n*(m+1)
                N = Np + Nu + Nv
                
                A, B = sp.lil_matrix((N, N)), np.zeros(N)

                # Équation de continuité (Pression)
                for j in range(m):
                    for i in range(n):
                        K = idx_P(i, j, n)
                        if i == 0:
                            A[K, K], B[K] = 1.0, P1
                        elif i == n - 1:
                            A[K, K], B[K] = 1.0, P2
                        elif j == 0:
                            A[K, K], A[K, idx_P(i, j + 1, n)] = -1.0, 1.0
                        elif j == m - 1:
                            A[K, K], A[K, idx_P(i, j - 1, n)] = 1.0, -1.0
                        else:
                            A[K, idx_U(i + 1, j, n)] = 1.0/hx
                            A[K, idx_U(i, j, n)] = -1.0/hx
                            A[K, idx_V(i, j + 1, n, m)] = 1.0/hy
                            A[K, idx_V(i, j, n, m)] = -1.0/hy

                # Équation de quantité de mouvement (Vitesse U)
                for j in range(m):
                    for i in range(n + 1):
                        K = idx_U(i, j, n)
                        if i == 0:
                            A[K, K], A[K, idx_U(i + 1, j, n)] = -1.0, 1.0
                        elif i == n:
                            A[K, K], A[K, idx_U(i - 1, j, n)] = 1.0, -1.0
                        elif j == 0 or j == m - 1:
                            A[K, K], B[K] = 1.0, 0.0
                        else:
                            A[K, idx_P(i, j, n)] = -1.0/hx
                            A[K, idx_P(i - 1, j, n)] = 1.0/hx
                            A[K, K] = -2.0 * C * (1.0/hx**2 + 1.0/hy**2)
                            A[K, idx_U(i + 1, j, n)] = C/hx**2
                            A[K, idx_U(i - 1, j, n)] = C/hx**2
                            A[K, idx_U(i, j + 1, n)] = C/hy**2
                            A[K, idx_U(i, j - 1, n)] = C/hy**2

                # Équation de quantité de mouvement (Vitesse V)
                for j in range(m + 1):
                    for i in range(n):
                        K = idx_V(i, j, n, m)
                        if j == 0 or j == m:
                            A[K, K], B[K] = 1.0, 0.0
                        elif i == 0:
                            A[K, K], B[K] = 1.0, 0.0
                        elif i == n - 1:
                            A[K, K], A[K, idx_V(i - 1, j, n, m)] = 1.0, -1.0
                        else:
                            A[K, idx_P(i, j, n)] = -1.0/hy
                            A[K, idx_P(i, j - 1, n)] = 1.0/hy
                            A[K, K] = -2.0 * C * (1.0/hx**2 + 1.0/hy**2)
                            A[K, idx_V(i + 1, j, n, m)] = C/hx**2
                            A[K, idx_V(i - 1, j, n, m)] = C/hx**2
                            A[K, idx_V(i, j + 1, n, m)] = C/hy**2
                            A[K, idx_V(i, j - 1, n, m)] = C/hy**2
                            
                return A.tocsr(), B

            A, B = matrixes(n, m, C, hx, hy, P1, P2)
            S = spsolve(A, B)

            # Extraction des champs
            P_vecteur = S[:Np]
            U_vecteur = S[Np:Np+Nu] 
            V_vecteur = S[Np+Nu:]

            P_grille_stag = P_vecteur.reshape((m, n))
            U_grille_stag = U_vecteur.reshape((m, n + 1))
            V_grille_stag = V_vecteur.reshape((m + 1, n))

            P_grille = P_grille_stag
            Ux_grille = (U_grille_stag[:, :-1] + U_grille_stag[:, 1:]) / 2.0
            Uy_grille = (V_grille_stag[:-1, :] + V_grille_stag[1:, :]) / 2.0
            
            self.after(0, self.afficher_graphique_tube_rigide, P_grille, Ux_grille, Uy_grille, L, l_val)
            
        except Exception as e:
            self.after(0, self.afficher_erreur, f"Erreur Tube Rigide: {str(e)}")
            
    def afficher_graphique_tube_rigide(self, P_grille, Ux_grille, Uy_grille, L, l_val):
        for widget in self.rigide_figure_frame.winfo_children():
            widget.destroy()
            
        # Créer un notebook pour organiser les graphiques
        notebook = ttk.Notebook(self.rigide_figure_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Graphique 1: Pression
        frame1 = ttk.Frame(notebook)
        notebook.add(frame1, text="Pression")
        
        fig1 = Figure(figsize=(6, 4))
        ax1 = fig1.add_subplot(111)
        im1 = ax1.imshow(P_grille, extent=[0, L, 0, l_val], origin='lower', aspect='auto', cmap='viridis')
        fig1.colorbar(im1, ax=ax1, label='Pression (Pa)')
        ax1.set_xlabel('Position x (m)')
        ax1.set_ylabel('Position y (m)')
        ax1.set_title('Champ de Pression', fontsize=12)
        
        canvas1 = FigureCanvasTkAgg(fig1, frame1)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Graphique 2: Vitesse horizontale
        frame2 = ttk.Frame(notebook)
        notebook.add(frame2, text="Vitesse Ux")
        
        fig2 = Figure(figsize=(6, 4))
        ax2 = fig2.add_subplot(111)
        im2 = ax2.imshow(Ux_grille, extent=[0, L, 0, l_val], origin='lower', aspect='auto', cmap='coolwarm')
        fig2.colorbar(im2, ax=ax2, label='Vitesse Ux (m/s)')
        ax2.set_xlabel('Position x (m)')
        ax2.set_ylabel('Position y (m)')
        ax2.set_title('Vitesse Horizontale', fontsize=12)
        
        canvas2 = FigureCanvasTkAgg(fig2, frame2)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Graphique 3: Vitesse verticale
        frame3 = ttk.Frame(notebook)
        notebook.add(frame3, text="Vitesse Uy")
        
        fig3 = Figure(figsize=(6, 4))
        ax3 = fig3.add_subplot(111)
        im3 = ax3.imshow(Uy_grille, extent=[0, L, 0, l_val], origin='lower', aspect='auto', cmap='coolwarm')
        fig3.colorbar(im3, ax=ax3, label='Vitesse Uy (m/s)')
        ax3.set_xlabel('Position x (m)')
        ax3.set_ylabel('Position y (m)')
        ax3.set_title('Vitesse Verticale', fontsize=12)
        
        canvas3 = FigureCanvasTkAgg(fig3, frame3)
        canvas3.draw()
        canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Graphique 4: Champ de vitesse combiné
        frame4 = ttk.Frame(notebook)
        notebook.add(frame4, text="Champ Vitesse")
        
        fig4 = Figure(figsize=(6, 4))
        ax4 = fig4.add_subplot(111)
        
        # Afficher la pression en arrière-plan
        im4 = ax4.imshow(P_grille, extent=[0, L, 0, l_val], origin='lower', aspect='auto', cmap='viridis', alpha=0.7)
        fig4.colorbar(im4, ax=ax4, label='Pression (Pa)')
        
        # Ajouter les vecteurs vitesse
        n_points = min(15, P_grille.shape[1]//8, P_grille.shape[0]//8)
        x = np.linspace(0, L, P_grille.shape[1])[::n_points]
        y = np.linspace(0, l_val, P_grille.shape[0])[::n_points]
        X, Y = np.meshgrid(x, y)
        
        Ux_subsampled = Ux_grille[::n_points, ::n_points]
        Uy_subsampled = Uy_grille[::n_points, ::n_points]
        
        # Calculer la magnitude pour l'échelle des vecteurs
        magnitude = np.sqrt(Ux_subsampled**2 + Uy_subsampled**2)
        scale = np.max(magnitude) * 10 if np.max(magnitude) > 0 else 1
        
        ax4.quiver(X, Y, Ux_subsampled, Uy_subsampled, magnitude, 
                  scale=scale, cmap='plasma', alpha=0.8)
        ax4.set_xlabel('Position x (m)')
        ax4.set_ylabel('Position y (m)')
        ax4.set_title('Champ de Vitesse + Pression', fontsize=12)
        
        canvas4 = FigureCanvasTkAgg(fig4, frame4)
        canvas4.draw()
        canvas4.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.status_var.set("Simulation Tube Rigide terminée")
        
    def lancer_geometrie_variable(self):
        self.status_var.set("Simulation Géométrie Variable en cours...")
        
        geom_type = self.geom_type.get()
        n = int(self.geom_n.get())
        m = int(self.geom_m.get())
        L = self.geom_L.get()
        l_val = self.geom_l.get()
        C = self.geom_C.get()
        P1 = self.geom_P1.get()
        P2 = self.geom_P2.get()
        
        thread = threading.Thread(target=self.executer_geometrie_variable, 
                                args=(geom_type, n, m, L, l_val, C, P1, P2))
        thread.daemon = True
        thread.start()
        
    def executer_geometrie_variable(self, geom_type, n, m, L, l_val, C, P1, P2):
        try:
            # Code 4 adapté selon le type
            hx, hy = L/n, l_val/m 
            y_centre = l_val / 2.0
            
            if geom_type == "cylindrique":
                rayon_entree, rayon_sortie = 3.0e-2, 2.0e-2
                R = np.linspace(rayon_entree, rayon_sortie, n + 1)
            else:  # aléatoire
                rayon_moyen = 2.5e-2
                amplitude_aleatoire = 2.5e-2
                taille_lissage = 8
                bruit_brut = (np.random.rand(n + 1) - 0.5) * 2 * amplitude_aleatoire
                kernel = np.ones(taille_lissage) / taille_lissage
                bruit_lisse = np.convolve(bruit_brut, kernel, mode='same')
                R = np.clip(rayon_moyen + bruit_lisse, 0.01 * l_val, y_centre * 0.95)
            
            # Le reste du code 4 reste identique
            Np, Nu, Nv = n*m, (n+1)*m, n*(m+1)
            N = Np + Nu + Nv

            def idx_P(i, j, n): return j * n + i 
            def idx_U(i, j, n): return j * (n + 1) + i + Np
            def idx_V(i, j, n, m): return j * n + i + Np + Nu

            def matrixes(n, m, C, hx, hy, P1, P2, R, l):
                Np, Nu, Nv = n*m, (n+1)*m, n*(m+1)
                N = Np + Nu + Nv
                A, B = sp.lil_matrix((N, N)), np.zeros(N)
                y_centre = l / 2.0

                # Équation de continuité (Pression)
                for j in range(m):
                    for i in range(n):
                        K, y_phys = idx_P(i, j, n), (j + 0.5) * hy
                        R_local, dist = (R[i] + R[i+1]) / 2.0, abs(y_phys - y_centre)

                        if dist > R_local:
                            if y_phys < y_centre:
                                A[K, K], A[K, idx_P(i, j + 1, n)] = -1.0, 1.0
                            else:
                                A[K, K], A[K, idx_P(i, j - 1, n)] = 1.0, -1.0
                        elif i == 0:
                            A[K, K], B[K] = 1.0, P1
                        elif i == n - 1:
                            A[K, K], B[K] = 1.0, P2
                        else:
                            A[K, idx_U(i + 1, j, n)] = 1.0/hx
                            A[K, idx_U(i, j, n)] = -1.0/hx
                            A[K, idx_V(i, j + 1, n, m)] = 1.0/hy
                            A[K, idx_V(i, j, n, m)] = -1.0/hy

                # Équation de quantité de mouvement (Vitesse U)
                for j in range(m):
                    for i in range(n + 1):
                        K, y_phys = idx_U(i, j, n), (j + 0.5) * hy
                        R_local, dist = R[i], abs(y_phys - y_centre)

                        if dist > R_local:
                            A[K, K], B[K] = 1.0, 0.0
                        elif i == 0:
                            A[K, K], A[K, idx_U(i + 1, j, n)] = -1.0, 1.0
                        elif i == n:
                            A[K, K], A[K, idx_U(i - 1, j, n)] = 1.0, -1.0
                        else:
                            A[K, idx_P(i, j, n)] = -1.0/hx
                            A[K, idx_P(i - 1, j, n)] = 1.0/hx
                            A[K, K] = -2.0 * C * (1.0/hx**2 + 1.0/hy**2)
                            A[K, idx_U(i + 1, j, n)] = C/hx**2
                            A[K, idx_U(i - 1, j, n)] = C/hx**2
                            A[K, idx_U(i, j + 1, n)] = C/hy**2
                            A[K, idx_U(i, j - 1, n)] = C/hy**2

                # Équation de quantité de mouvement (Vitesse V)
                for j in range(m + 1):
                    for i in range(n):
                        K, y_phys = idx_V(i, j, n, m), j * hy
                        R_local, dist = (R[i] + R[i+1]) / 2.0, abs(y_phys - y_centre)

                        if j == 0 or j == m or dist > R_local:
                            A[K, K], B[K] = 1.0, 0.0
                        elif i == 0:
                            A[K, K], B[K] = 1.0, 0.0
                        elif i == n - 1:
                            A[K, K], A[K, idx_V(i - 1, j, n, m)] = 1.0, -1.0
                        else:
                            A[K, idx_P(i, j, n)] = -1.0/hy
                            A[K, idx_P(i, j - 1, n)] = 1.0/hy
                            A[K, K] = -2.0 * C * (1.0/hx**2 + 1.0/hy**2)
                            A[K, idx_V(i + 1, j, n, m)] = C/hx**2
                            A[K, idx_V(i - 1, j, n, m)] = C/hx**2
                            A[K, idx_V(i, j + 1, n, m)] = C/hy**2
                            A[K, idx_V(i, j - 1, n, m)] = C/hy**2
                            
                return A.tocsr(), B

            A, B = matrixes(n, m, C, hx, hy, P1, P2, R, l_val)
            S = spsolve(A, B)

            # Extraction des champs
            P_vecteur = S[:Np]
            U_vecteur = S[Np:Np+Nu] 
            V_vecteur = S[Np+Nu:]

            P_grille = P_vecteur.reshape((m, n))
            Ux_grille = (U_vecteur.reshape((m, n + 1))[:, :-1] + U_vecteur.reshape((m, n + 1))[:, 1:]) / 2.0
            Uy_grille = (V_vecteur.reshape((m + 1, n))[:-1, :] + V_vecteur.reshape((m + 1, n))[1:, :]) / 2.0

            # Masquage des régions solides avec bordures noires
            x_coords = np.linspace(hx/2.0, L - hx/2.0, n)
            y_coords = np.linspace(hy/2.0, l_val - hy/2.0, m)
            
            R_local_p = (R[:-1] + R[1:]) / 2.0 
            dist_au_centre_p = abs(y_coords.reshape(m, 1) - l_val/2.0)
            masque_mur = dist_au_centre_p > R_local_p.reshape(1, n)

            # Appliquer le masque avec bordures noires
            P_grille_plot = np.where(masque_mur, np.nan, P_grille)
            Ux_grille_plot = np.where(masque_mur, np.nan, Ux_grille)
            Uy_grille_plot = np.where(masque_mur, np.nan, Uy_grille)
            
            self.after(0, self.afficher_graphique_geometrie_variable, 
                      P_grille_plot, Ux_grille_plot, Uy_grille_plot, L, l_val, R, geom_type)
            
        except Exception as e:
            self.after(0, self.afficher_erreur, f"Erreur Géométrie Variable: {str(e)}")
            
    def afficher_graphique_geometrie_variable(self, P_grille, Ux_grille, Uy_grille, L, l_val, R, geom_type):
        for widget in self.geom_figure_frame.winfo_children():
            widget.destroy()
            
        # Notebook pour organiser les graphiques
        notebook = ttk.Notebook(self.geom_figure_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Graphique 1: Pression
        frame1 = ttk.Frame(notebook)
        notebook.add(frame1, text="Pression")
        
        fig1 = Figure(figsize=(6, 4))
        ax1 = fig1.add_subplot(111)
        cmap_pression = copy.copy(plt.cm.viridis)
        cmap_pression.set_bad('black', 1.0)  # Bordures noires
        im1 = ax1.imshow(P_grille, extent=[0, L, 0, l_val], origin='lower', aspect='auto', cmap=cmap_pression)
        fig1.colorbar(im1, ax=ax1, label='Pression (Pa)')
        ax1.set_xlabel('Position x (m)')
        ax1.set_ylabel('Position y (m)')
        ax1.set_title(f'Pression - Géométrie {geom_type}', fontsize=12)
        
        canvas1 = FigureCanvasTkAgg(fig1, frame1)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Graphique 2: Vitesse horizontale
        frame2 = ttk.Frame(notebook)
        notebook.add(frame2, text="Vitesse Ux")
        
        fig2 = Figure(figsize=(6, 4))
        ax2 = fig2.add_subplot(111)
        cmap_vitesse = copy.copy(plt.cm.coolwarm)
        cmap_vitesse.set_bad('black', 1.0)  # Bordures noires
        im2 = ax2.imshow(Ux_grille, extent=[0, L, 0, l_val], origin='lower', aspect='auto', cmap=cmap_vitesse)
        fig2.colorbar(im2, ax=ax2, label='Vitesse Ux (m/s)')
        ax2.set_xlabel('Position x (m)')
        ax2.set_ylabel('Position y (m)')
        ax2.set_title('Vitesse Horizontale', fontsize=12)
        
        canvas2 = FigureCanvasTkAgg(fig2, frame2)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Graphique 3: Vitesse verticale
        frame3 = ttk.Frame(notebook)
        notebook.add(frame3, text="Vitesse Uy")
        
        fig3 = Figure(figsize=(6, 4))
        ax3 = fig3.add_subplot(111)
        cmap_vitesse = copy.copy(plt.cm.coolwarm)
        cmap_vitesse.set_bad('black', 1.0)  # Bordures noires
        im3 = ax3.imshow(Uy_grille, extent=[0, L, 0, l_val], origin='lower', aspect='auto', cmap=cmap_vitesse)
        fig3.colorbar(im3, ax=ax3, label='Vitesse Uy (m/s)')
        ax3.set_xlabel('Position x (m)')
        ax3.set_ylabel('Position y (m)')
        ax3.set_title('Vitesse Verticale', fontsize=12)
        
        canvas3 = FigureCanvasTkAgg(fig3, frame3)
        canvas3.draw()
        canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Graphique 4: Champ de vitesse combiné
        frame4 = ttk.Frame(notebook)
        notebook.add(frame4, text="Champ Vitesse")
        
        fig4 = Figure(figsize=(6, 4))
        ax4 = fig4.add_subplot(111)
        
        # Afficher la pression en arrière-plan
        cmap_pression = copy.copy(plt.cm.viridis)
        cmap_pression.set_bad('black', 1.0)
        im4 = ax4.imshow(P_grille, extent=[0, L, 0, l_val], origin='lower', aspect='auto', cmap=cmap_pression, alpha=0.7)
        fig4.colorbar(im4, ax=ax4, label='Pression (Pa)')
        
        # Ajouter les vecteurs vitesse
        n_points = min(12, P_grille.shape[1]//10, P_grille.shape[0]//10)
        x = np.linspace(0, L, P_grille.shape[1])[::n_points]
        y = np.linspace(0, l_val, P_grille.shape[0])[::n_points]
        X, Y = np.meshgrid(x, y)
        
        Ux_subsampled = Ux_grille[::n_points, ::n_points]
        Uy_subsampled = Uy_grille[::n_points, ::n_points]
        
        # Masquer les vecteurs dans les régions solides
        mask = ~np.isnan(Ux_subsampled)
        
        # Calculer la magnitude pour l'échelle des vecteurs
        magnitude = np.sqrt(Ux_subsampled**2 + Uy_subsampled**2)
        scale = np.nanmax(magnitude) * 8 if not np.all(np.isnan(magnitude)) else 1
        
        ax4.quiver(X[mask], Y[mask], Ux_subsampled[mask], Uy_subsampled[mask], 
                  magnitude[mask], scale=scale, cmap='plasma', alpha=0.8)
        ax4.set_xlabel('Position x (m)')
        ax4.set_ylabel('Position y (m)')
        ax4.set_title('Champ de Vitesse + Pression', fontsize=12)
        
        canvas4 = FigureCanvasTkAgg(fig4, frame4)
        canvas4.draw()
        canvas4.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Graphique 5: Profil géométrique
        frame5 = ttk.Frame(notebook)
        notebook.add(frame5, text="Profil Géométrique")
        
        fig5 = Figure(figsize=(6, 4))
        ax5 = fig5.add_subplot(111)
        x_positions = np.linspace(0, L, len(R))
        ax5.plot(x_positions, R, 'b-', linewidth=2, label='Profil du tube')
        ax5.fill_between(x_positions, l_val/2 - R, l_val/2 + R, alpha=0.3, color='blue')
        ax5.set_xlabel('Position x (m)')
        ax5.set_ylabel('Rayon (m)')
        ax5.set_title('Profil Géométrique', fontsize=12)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        canvas5 = FigureCanvasTkAgg(fig5, frame5)
        canvas5.draw()
        canvas5.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.status_var.set(f"Simulation Géométrie {geom_type} terminée")

    def lancer_tube_souple(self):
        self.status_var.set("Simulation Tube Souple en cours...")
        
        # Récupération des paramètres
        R0 = self.souple_R0.get()
        alpha0_val = self.souple_alpha0.get()
        n1_const = self.souple_n1.get()
        n2_const = self.souple_n2.get()
        P1_const = self.souple_P1.get()
        P2_const = self.souple_P2.get()
        P0 = self.souple_P0.get()
        L = self.souple_L.get()
        P_entree = self.souple_P_entree.get()
        P_sortie = self.souple_P_sortie.get()
        n = int(self.souple_n.get())
        m = int(self.souple_m.get())
        facteur_zoom_visuel = self.souple_zoom.get()
        C = self.souple_C.get()
        max_iterations = int(self.souple_iter_max.get())
        tolerance = self.souple_tolerance.get()
        
        thread = threading.Thread(target=self.executer_tube_souple, 
                                args=(R0, alpha0_val, n1_const, n2_const, P1_const, P2_const, P0,
                                      L, P_entree, P_sortie, n, m, facteur_zoom_visuel, C, 
                                      max_iterations, tolerance))
        thread.daemon = True
        thread.start()

    def executer_tube_souple(self, R0, alpha0_val, n1_const, n2_const, P1_const, P2_const, P0,
                           L, P_entree, P_sortie, n_base, m_base, facteur_zoom_visuel, C, 
                           max_iterations, tolerance):
        try:
            # Code 5 adapté pour l'interface
            R_repos = R0 * np.sqrt(alpha0_val)
            
            n = n_base
            m = int(m_base * facteur_zoom_visuel)
            
            l = R0 * 2 * facteur_zoom_visuel
            hx = L/n
            hy = l/m
            
            Np = n*m
            Nu = (n+1)*m
            Nv = n*(m+1)
            N = Np + Nu + Nv
            
            y_centre = l / 2.0
            R = np.full(n + 1, R_repos)

            def idx_P(i, j, n):
                return (j * n + i) 
            
            def idx_U(i, j, n):
                return (j * (n + 1) + i) + Np
            
            def idx_V(i, j, n, m):
                return (j * n + i) + Np + Nu

            def D_tube(P, P_pl, P1, P2, n1, n2, alpha0, Dmax):
                P_trans = P - P_pl

                if P_trans <= 0:
                    terme = 1 - (P_trans / P1)
                    terme = max(terme, 1e-9)
                    D = Dmax * np.sqrt(alpha0 * terme ** (-n1))
                else:
                    terme = 1 - (P_trans / P2)
                    terme = max(terme, 1e-9)
                    D = Dmax * np.sqrt(1 - (1 - alpha0) * terme ** (-n2))
                
                return D

            def matrixes(n, m, C, hx, hy, P_entree_BC, P_sortie_BC, R_geom, l_geom):
                Np = n * m
                Nu = (n + 1) * m
                Nv = n * (m + 1)
                N = Np + Nu + Nv
                A = sp.lil_matrix((N, N))
                B = np.zeros(N)
                y_centre_geom = l_geom / 2.0

                for j in range(m):
                    for i in range(n):
                        K = idx_P(i, j, n)
                        y_phys = (j + 0.5) * hy
                        R_local = (R_geom[i] + R_geom[i+1]) / 2.0
                        dist_au_centre = abs(y_phys - y_centre_geom)
                        
                        if dist_au_centre > R_local:
                            if y_phys < y_centre_geom:
                                A[K, K] = -1.0
                                if j+1 < m: 
                                    A[K, idx_P(i, j + 1, n)] = 1.0
                            else:
                                A[K, K] = 1.0
                                if j-1 >= 0: 
                                    A[K, idx_P(i, j - 1, n)] = -1.0
                        elif i == 0:
                            A[K, K] = 1.0
                            B[K] = P_entree_BC
                        elif i == n - 1:
                            A[K, K] = 1.0
                            B[K] = P_sortie_BC
                        else:
                            A[K, idx_U(i + 1, j, n)] = 1.0 / hx
                            A[K, idx_U(i, j, n)] = -1.0 / hx
                            A[K, idx_V(i, j + 1, n, m)] = 1.0 / hy
                            A[K, idx_V(i, j, n, m)] = -1.0 / hy

                for j in range(m):
                    for i in range(n + 1):
                        K = idx_U(i, j, n)
                        y_phys = (j + 0.5) * hy
                        R_local = R_geom[i]
                        dist_au_centre = abs(y_phys - y_centre_geom)
                        
                        if dist_au_centre > R_local:
                            A[K, K] = 1.0
                        elif i == 0:
                            A[K, K] = -1.0
                            A[K, idx_U(i + 1, j, n)] = 1.0
                        elif i == n:
                            A[K, K] = 1.0
                            A[K, idx_U(i - 1, j, n)] = -1.0
                        else:
                            A[K, idx_P(i, j, n)] = -1.0 / hx
                            A[K, idx_P(i - 1, j, n)] = 1.0 / hx
                            A[K, K] = -2.0 * C * (1.0/hx**2 + 1.0/hy**2)
                            A[K, idx_U(i + 1, j, n)] = C / hx**2
                            A[K, idx_U(i - 1, j, n)] = C / hx**2
                            A[K, idx_U(i, j + 1, n)] = C / hy**2
                            A[K, idx_U(i, j - 1, n)] = C / hy**2

                for j in range(m + 1):
                    for i in range(n):
                        K = idx_V(i, j, n, m)
                        y_phys = j * hy
                        R_local = (R_geom[i] + R_geom[i+1]) / 2.0
                        dist_au_centre = abs(y_phys - y_centre_geom)
                        
                        if j == 0 or j == m or dist_au_centre > R_local:
                            A[K, K] = 1.0
                        elif i == 0:
                            A[K, K] = 1.0
                        elif i == n - 1:
                            A[K, K] = 1.0
                            A[K, idx_V(i - 1, j, n, m)] = -1.0
                        else:
                            A[K, idx_P(i, j, n)] = -1.0 / hy
                            A[K, idx_P(i, j - 1, n)] = 1.0 / hy
                            A[K, K] = -2.0 * C * (1.0/hx**2 + 1.0/hy**2)
                            A[K, idx_V(i + 1, j, n, m)] = C / hx**2
                            A[K, idx_V(i - 1, j, n, m)] = C / hx**2
                            A[K, idx_V(i, j + 1, n, m)] = C / hy**2
                            A[K, idx_V(i, j - 1, n, m)] = C / hy**2
                            
                return A.tocsr(), B

            iteration = 0
            changement = True
            R_history = [R.copy()]
            convergence_data = []
            S_final = None

            while iteration < max_iterations and changement:
                self.after(0, lambda: self.status_var.set(f"Tube Souple - Itération {iteration + 1}/{max_iterations}"))
                
                A, B = matrixes(n, m, C, hx, hy, P_entree, P_sortie, R, l)
                
                try:
                    S = spsolve(A, B)
                except Exception as e:
                    self.after(0, self.afficher_erreur, f"Erreur résolution itération {iteration}: {str(e)}")
                    break
                
                offset_P = 0
                offset_U = Np
                offset_V = Np + Nu
                P_vecteur = S[offset_P:offset_U]
                P_grille = P_vecteur.reshape((m, n))
                
                pression_au_mur = np.zeros(n)
                y_centre_phys = l / 2.0
                for i in range(n):
                    R_local_p = (R[i] + R[i+1]) / 2.0
                    y_mur_sup_phys = y_centre_phys + R_local_p
                    j_mur_idx = int((y_mur_sup_phys / hy) - 0.5 - 1e-6)
                    j_mur_idx = min(max(j_mur_idx, 0), m - 1)
                    pression_au_mur[i] = P_grille[j_mur_idx, i]

                R_ancien = R.copy()
                changement = False
                change_max = 0.0
                
                for i in range(n + 1):
                    if i == 0:
                        pression_interieure = pression_au_mur[0]
                    elif i == n:
                        pression_interieure = pression_au_mur[n-1]
                    else:
                        pression_interieure = (pression_au_mur[i-1] + pression_au_mur[i]) / 2.0
                    
                    R_new = D_tube(pression_interieure, P0, P1_const, P2_const, 
                                  n1_const, n2_const, alpha0_val, 2*R0) / 2.0
                    
                    R_new = R_ancien[i] + (R_new - R_ancien[i]) / (-np.log10(tolerance))
                    
                    change = abs(R_new - R_ancien[i])
                    if change > tolerance:
                        changement = True
                    if change > change_max:
                        change_max = change

                    R[i] = R_new

                R_history.append(R.copy())
                convergence_data.append({
                    'iteration': iteration,
                    'changement_max': change_max,
                })
                
                iteration += 1
                S_final = S

            # Extraction des résultats finaux
            if S_final is not None:
                P_vecteur = S_final[offset_P:offset_U]
                U_vecteur = S_final[offset_U:offset_V]
                V_vecteur = S_final[offset_V:]

                P_grille_stag = P_vecteur.reshape((m, n))
                U_grille_stag = U_vecteur.reshape((m, n + 1))
                V_grille_stag = V_vecteur.reshape((m + 1, n))

                P_grille = P_grille_stag
                Ux_grille = (U_grille_stag[:, :-1] + U_grille_stag[:, 1:]) / 2.0
                Uy_grille = (V_grille_stag[:-1, :] + V_grille_stag[1:, :]) / 2.0

                # Préparation des données pour l'affichage
                donnees_affichage = {
                    'R_history': R_history,
                    'convergence_data': convergence_data,
                    'P_grille': P_grille,
                    'Ux_grille': Ux_grille,
                    'Uy_grille': Uy_grille,
                    'R_final': R,
                    'L': L,
                    'l': l,
                    'iteration_finale': iteration,
                    'hx': hx,
                    'hy': hy
                }
                
                self.after(0, self.afficher_graphique_tube_souple, donnees_affichage)
            else:
                self.after(0, self.afficher_erreur, "La simulation du tube souple n'a pas convergé")
                
        except Exception as e:
            self.after(0, self.afficher_erreur, f"Erreur Tube Souple: {str(e)}")

    def afficher_graphique_tube_souple(self, donnees):
        for widget in self.souple_figure_frame.winfo_children():
            widget.destroy()
            
        # Notebook pour organiser les graphiques
        notebook = ttk.Notebook(self.souple_figure_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Graphique 1: Convergence - FIGURE PLUS GRANDE POUR ÉVITER LA CONDENSATION
        frame1 = ttk.Frame(notebook)
        notebook.add(frame1, text="Convergence")
        
        # Créer une figure plus grande avec un meilleur espacement
        fig1 = Figure(figsize=(14, 10))
        fig1.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.92, wspace=0.3, hspace=0.4)
        
        # Évolution rayons - Subplot plus grand
        ax1 = fig1.add_subplot(2, 2, 1)
        iterations = range(len(donnees['R_history']))
        n = len(donnees['R_final']) - 1
        # Réduire le nombre de courbes affichées pour plus de clarté
        for i in range(0, n+1, max(1, (n+1)//5)):  # Afficher seulement 5 courbes au lieu de 10
            rayons_colonne = [donnees['R_history'][iter][i] for iter in iterations]
            ax1.plot(iterations, rayons_colonne, label=f'Colonne {i}', linewidth=1.5)
        ax1.set_xlabel('Itération', fontsize=10)
        ax1.set_ylabel('Rayon (m)', fontsize=10)
        ax1.set_title('Évolution rayons par colonne', fontsize=11, fontweight='bold')
        ax1.legend(fontsize=8, loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=9)
        
        # Profil rayon final - Subplot plus grand
        ax2 = fig1.add_subplot(2, 2, 2)
        x_positions = np.linspace(0, donnees['L'], len(donnees['R_final']))
        ax2.plot(x_positions * 100, donnees['R_final'] * 100, 'b-', linewidth=2, label='Rayon final')
        R0_val = np.max(donnees['R_final'])
        R_repos_val = np.mean(donnees['R_final'])
        ax2.axhline(y=R0_val * 100, color='r', linestyle='--', label='Rayon maximal', alpha=0.7)
        ax2.axhline(y=R_repos_val * 100, color='g', linestyle='--', label='Rayon repos', alpha=0.7)
        ax2.set_xlabel('Position x (cm)', fontsize=10)
        ax2.set_ylabel('Rayon (cm)', fontsize=10)
        ax2.set_title('Profil rayon final', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=9)
        
        # Statistiques rayons - Subplot plus grand
        ax3 = fig1.add_subplot(2, 2, 3)
        rayon_moyen = [np.mean(donnees['R_history'][iter]) for iter in iterations]
        rayon_min = [np.min(donnees['R_history'][iter]) for iter in iterations]
        rayon_max = [np.max(donnees['R_history'][iter]) for iter in iterations]
        ax3.plot(iterations, rayon_moyen, 'b-', linewidth=2, label='Moyen')
        ax3.plot(iterations, rayon_min, 'r--', linewidth=1.5, label='Minimum')
        ax3.plot(iterations, rayon_max, 'g--', linewidth=1.5, label='Maximum')
        ax3.set_xlabel('Itération', fontsize=10)
        ax3.set_ylabel('Rayon (m)', fontsize=10)
        ax3.set_title('Statistiques rayons', fontsize=11, fontweight='bold')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(labelsize=9)
        
        # Convergence - Subplot plus grand
        ax4 = fig1.add_subplot(2, 2, 4)
        changements = [data['changement_max'] for data in donnees['convergence_data'] if data['changement_max'] > 0]
        if changements:
            ax4.semilogy(range(1, len(changements)+1), changements, 'r-', linewidth=2)
        ax4.set_xlabel('Itération', fontsize=10)
        ax4.set_ylabel('Changement max (log)', fontsize=10)
        ax4.set_title('Convergence', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(labelsize=9)
        
        canvas1 = FigureCanvasTkAgg(fig1, frame1)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Graphique 2: Champ physique - FIGURE PLUS GRANDE POUR ÉVITER LA CONDENSATION
        frame2 = ttk.Frame(notebook)
        notebook.add(frame2, text="Champs Physiques")
        
        # Préparation des données pour l'affichage
        x_coords = np.linspace(donnees['hx']/2.0, donnees['L'] - donnees['hx']/2.0, donnees['P_grille'].shape[1])
        y_coords = np.linspace(donnees['hy']/2.0, donnees['l'] - donnees['hy']/2.0, donnees['P_grille'].shape[0])
        
        R_local_p_centres = (donnees['R_final'][:-1] + donnees['R_final'][1:]) / 2.0
        dist_au_centre_p = abs(y_coords.reshape(len(y_coords), 1) - donnees['l']/2.0)
        masque_mur = dist_au_centre_p > R_local_p_centres.reshape(1, len(R_local_p_centres))
        
        P_grille_plot = np.where(masque_mur, np.nan, donnees['P_grille'])
        Ux_grille_plot = np.where(masque_mur, np.nan, donnees['Ux_grille'])
        Uy_grille_plot = np.where(masque_mur, np.nan, donnees['Uy_grille'])
        
        # Créer une figure plus grande avec un meilleur espacement
        fig2 = Figure(figsize=(14, 10))
        fig2.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.92, wspace=0.3, hspace=0.4)
        
        # Pression - Subplot plus grand
        ax1 = fig2.add_subplot(2, 2, 1)
        cmap_pression = copy.copy(plt.cm.viridis)
        cmap_pression.set_bad('black')
        im1 = ax1.imshow(P_grille_plot, extent=[0, donnees['L']*100, 0, donnees['l']*100], 
                        origin='lower', aspect='auto', cmap=cmap_pression)
        cbar1 = fig2.colorbar(im1, ax=ax1, label='Pression (Pa)')
        cbar1.ax.tick_params(labelsize=9)
        ax1.set_xlabel('Position x (cm)', fontsize=10)
        ax1.set_ylabel('Position y (cm)', fontsize=10)
        ax1.set_title('Champ de Pression', fontsize=11, fontweight='bold')
        ax1.tick_params(labelsize=9)
        
        # Vitesse horizontale - Subplot plus grand
        ax2 = fig2.add_subplot(2, 2, 2)
        cmap_vitesse = copy.copy(plt.cm.coolwarm)
        cmap_vitesse.set_bad('black')
        im2 = ax2.imshow(Ux_grille_plot, extent=[0, donnees['L']*100, 0, donnees['l']*100], 
                        origin='lower', aspect='auto', cmap=cmap_vitesse)
        cbar2 = fig2.colorbar(im2, ax=ax2, label='Vitesse Ux (m/s)')
        cbar2.ax.tick_params(labelsize=9)
        ax2.set_xlabel('Position x (cm)', fontsize=10)
        ax2.set_ylabel('Position y (cm)', fontsize=10)
        ax2.set_title('Vitesse Horizontale', fontsize=11, fontweight='bold')
        ax2.tick_params(labelsize=9)
        
        # Pression + vecteurs vitesse - Subplot plus grand
        ax3 = fig2.add_subplot(2, 2, 3)
        im3 = ax3.imshow(P_grille_plot, extent=[0, donnees['L']*100, 0, donnees['l']*100], 
                        origin='lower', aspect='auto', cmap=cmap_pression)
        cbar3 = fig2.colorbar(im3, ax=ax3, label='Pression (Pa)')
        cbar3.ax.tick_params(labelsize=9)
        
        # Ajouter les vecteurs vitesse avec un espacement adapté
        pas = max(1, donnees['P_grille'].shape[1] // 12)  # Réduire le nombre de vecteurs pour plus de clarté
        X, Y = np.meshgrid(x_coords, y_coords)
        # Filtrer les zones où il y a des données valides
        mask_vectors = ~np.isnan(Ux_grille_plot[::pas, ::pas])
        ax3.quiver(X[::pas, ::pas][mask_vectors] * 100, Y[::pas, ::pas][mask_vectors] * 100, 
                  Ux_grille_plot[::pas, ::pas][mask_vectors], Uy_grille_plot[::pas, ::pas][mask_vectors], 
                  color='white', scale=np.nanmax(Ux_grille_plot)*25, alpha=0.7)
        ax3.set_xlabel('Position x (cm)', fontsize=10)
        ax3.set_ylabel('Position y (cm)', fontsize=10)
        ax3.set_title('Pression et vitesse', fontsize=11, fontweight='bold')
        ax3.tick_params(labelsize=9)
        
        # Profil géométrique final - Subplot plus grand
        ax4 = fig2.add_subplot(2, 2, 4)
        x_positions = np.linspace(0, donnees['L'], len(donnees['R_final']))
        ax4.plot(x_positions * 100, donnees['R_final'] * 100, 'b-', linewidth=2, label='Rayon final')
        ax4.fill_between(x_positions * 100, donnees['l']/2*100 - donnees['R_final']*100, 
                        donnees['l']/2*100 + donnees['R_final']*100, alpha=0.3, color='blue')
        ax4.set_xlabel('Position x (cm)', fontsize=10)
        ax4.set_ylabel('Rayon (cm)', fontsize=10)
        ax4.set_title('Profil géométrique final', fontsize=11, fontweight='bold')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(labelsize=9)
        
        canvas2 = FigureCanvasTkAgg(fig2, frame2)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.status_var.set(f"Simulation Tube Souple terminée ({donnees['iteration_finale']} itérations)")

    def lancer_toutes_generations(self):
        self.status_var.set("Simulation de toutes les générations en cours...")
        
        # Réinitialiser la barre de progression
        self.gen_progress['value'] = 0
        self.gen_progress_label['text'] = "0/17 générations"
        
        # Récupération des paramètres
        cas = self.gen_type.get()
        max_iterations = int(self.gen_iter_max.get())
        tolerance = self.gen_tolerance.get()
        C = self.gen_C.get()
        
        thread = threading.Thread(target=self.executer_toutes_generations, 
                                args=(cas, max_iterations, tolerance, C))
        thread.daemon = True
        thread.start()

    def executer_toutes_generations(self, cas, max_iterations, tolerance, C):
        try:
            # Données des générations bronchiques
            generations_data = {
                'gen': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                'longueur': [0.1200, 0.0476, 0.0190, 0.0076, 0.0127, 0.0107, 0.0090, 0.0076, 0.0064, 0.0054, 0.0047, 0.0039, 0.0033, 0.0027, 0.0023, 0.0020, 0.0017],
                'R0': [0.00868, 0.00614, 0.00489, 0.00373, 0.00289, 0.00225, 0.00175, 0.00138, 0.00108, 0.00088, 0.00068, 0.00055, 0.00047, 0.00042, 0.00037, 0.00033, 0.00029],
                'alpha0': [0.882, 0.882, 0.686, 0.546, 0.428, 0.337, 0.265, 0.208, 0.164, 0.129, 0.102, 0.080, 0.063, 0.049, 0.039, 0.031, 0.024],
                'n1': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                'n2': [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 8, 8, 8, 7, 7],
                'P1': [8009, 2942, 1345, 687, 428, 268, 187, 131, 94, 70, 53, 39, 29, 22, 17, 13, 10],
                'P2': [-10715, -3936, -6158, -5707, -5709, -5282, -5192, -4993, -4796, -4712, -4635, -4009, -3444, -3392, -3301, -2879, -2814]
            }

            # Pressions pour l'expiration forcée
            pressions_expiration_forcee = {
                0: {'entree': -490, 'sortie': -480},
                1: {'entree': -400, 'sortie': -390},
                2: {'entree': -310, 'sortie': -300},
                3: {'entree': -220, 'sortie': -210},
                4: {'entree': -147, 'sortie': -137},
                5: {'entree': 78, 'sortie': 68},
                6: {'entree': 128, 'sortie': 118},
                7: {'entree': 178, 'sortie': 168},
                8: {'entree': 228, 'sortie': 218},
                9: {'entree': 278, 'sortie': 268},
                10: {'entree': 490.5, 'sortie': 451.3},
                11: {'entree': 350, 'sortie': 340},
                12: {'entree': 400, 'sortie': 390},
                13: {'entree': 450, 'sortie': 440},
                14: {'entree': 500, 'sortie': 490},
                15: {'entree': 550, 'sortie': 540},
                16: {'entree': 588, 'sortie': 578}
            }

            resultats = {}
            
            # Simulation pour chaque génération
            for gen_index in range(len(generations_data['gen'])):
                gen = generations_data['gen'][gen_index]
                
                # Mise à jour de la progression
                self.after(0, self.mettre_a_jour_progression, gen_index, len(generations_data['gen']))
                
                try:
                    # Paramètres de la génération
                    L = generations_data['longueur'][gen_index]
                    R0 = generations_data['R0'][gen_index]
                    alpha0_val = generations_data['alpha0'][gen_index]
                    n1_const = generations_data['n1'][gen_index]
                    n2_const = generations_data['n2'][gen_index]
                    P1_const = generations_data['P1'][gen_index]
                    P2_const = generations_data['P2'][gen_index]
                    
                    # Calcul du rayon au repos
                    R_repos = R0 * np.sqrt(alpha0_val)
                    
                    # Pressions selon le cas
                    if cas == 'expiration_forcee':
                        P_entree = pressions_expiration_forcee[gen]['entree']
                        P_sortie = pressions_expiration_forcee[gen]['sortie']
                    elif cas == 'inspiration':
                        P_entree = 49.0
                        P_sortie = 20.0
                    else:  # repos
                        P_entree = 0.0
                        P_sortie = 0.0
                    
                    P0 = 0.0  # Pression externe

                    # Paramètres grille numérique (adaptés à la taille de la génération)
                    n_base = max(50, int(L * 10000))
                    m_base = max(50, int(R0 * 20000))
                    
                    n = min(n_base, 150)
                    m = min(m_base, 150)
                    
                    # Dimensions domaine calcul
                    l = R0 * 4
                    hx = L/n
                    hy = l/m
                    
                    # Tailles systèmes
                    Np = n*m
                    Nu = (n+1)*m
                    Nv = n*(m+1)
                    N = Np + Nu + Nv
                    
                    # Géométrie initiale
                    y_centre = l / 2.0
                    R = np.full(n + 1, R_repos)
                    
                    # Fonctions d'indexation
                    def idx_P(i, j, n):
                        return (j * n + i) 
                    
                    def idx_U(i, j, n):
                        return (j * (n + 1) + i)
                    
                    def idx_V(i, j, n, m):
                        return (j * n + i)
                    
                    def D_tube(P, P_pl, P1, P2, n1, n2, alpha0, Dmax):
                        P_trans = P - P_pl

                        if P_trans <= 0:
                            terme = 1 - (P_trans / P1)
                            terme = max(terme, 1e-9)
                            D = Dmax * np.sqrt(alpha0 * terme ** (-n1))
                        else:
                            terme = 1 - (P_trans / P2)
                            terme = max(terme, 1e-9)
                            D = Dmax * np.sqrt(1 - (1 - alpha0) * terme ** (-n2))
                        
                        return D
                    
                    def matrixes(n, m, C, hx, hy, P_entree_BC, P_sortie_BC, R_geom, l_geom, Np, Nu, Nv):
                        N = Np + Nu + Nv
                        A = sp.lil_matrix((N, N))
                        B = np.zeros(N)
                        y_centre_geom = l_geom / 2.0

                        # Équations de continuité (pression)
                        for j in range(m):
                            for i in range(n):
                                K = idx_P(i, j, n) 
                                y_phys = (j + 0.5) * hy
                                R_local = (R_geom[i] + R_geom[i+1]) / 2.0
                                dist_au_centre = abs(y_phys - y_centre_geom)
                                
                                if dist_au_centre > R_local:
                                    if y_phys < y_centre_geom:
                                        A[K, K] = -1.0
                                        if j+1 < m: 
                                            A[K, idx_P(i, j + 1, n)] = 1.0
                                    else:
                                        A[K, K] = 1.0
                                        if j-1 >= 0: 
                                            A[K, idx_P(i, j - 1, n)] = -1.0
                                elif i == 0:
                                    A[K, K] = 1.0
                                    B[K] = P_entree_BC
                                elif i == n - 1:
                                    A[K, K] = 1.0
                                    B[K] = P_sortie_BC
                                else:
                                    A[K, idx_U(i + 1, j, n) + Np] = 1.0 / hx
                                    A[K, idx_U(i, j, n) + Np] = -1.0 / hx
                                    A[K, idx_V(i, j + 1, n, m) + Np + Nu] = 1.0 / hy
                                    A[K, idx_V(i, j, n, m) + Np + Nu] = -1.0 / hy

                        # Équations quantité mouvement U
                        for j in range(m):
                            for i in range(n + 1):
                                K = idx_U(i, j, n) + Np
                                y_phys = (j + 0.5) * hy
                                R_local = R_geom[i]
                                dist_au_centre = abs(y_phys - y_centre_geom)
                                
                                if dist_au_centre > R_local:
                                    A[K, K] = 1.0
                                elif i == 0:
                                    A[K, K] = -1.0
                                    A[K, idx_U(i + 1, j, n) + Np] = 1.0
                                elif i == n:
                                    A[K, K] = 1.0
                                    A[K, idx_U(i - 1, j, n) + Np] = -1.0
                                else:
                                    A[K, idx_P(i, j, n)] = -1.0 / hx
                                    A[K, idx_P(i - 1, j, n)] = 1.0 / hx
                                    A[K, K] = -2.0 * C * (1.0/hx**2 + 1.0/hy**2)
                                    A[K, idx_U(i + 1, j, n) + Np] = C / hx**2
                                    A[K, idx_U(i - 1, j, n) + Np] = C / hx**2
                                    A[K, idx_U(i, j + 1, n) + Np] = C / hy**2
                                    A[K, idx_U(i, j - 1, n) + Np] = C / hy**2

                        # Équations quantité mouvement V
                        for j in range(m + 1):
                            for i in range(n):
                                K = idx_V(i, j, n, m) + Np + Nu
                                y_phys = j * hy
                                R_local = (R_geom[i] + R_geom[i+1]) / 2.0
                                dist_au_centre = abs(y_phys - y_centre_geom)
                                
                                if j == 0 or j == m or dist_au_centre > R_local:
                                    A[K, K] = 1.0
                                elif i == 0:
                                    A[K, K] = 1.0
                                elif i == n - 1:
                                    A[K, K] = 1.0
                                    A[K, idx_V(i - 1, j, n, m) + Np + Nu] = -1.0
                                else:
                                    A[K, idx_P(i, j, n)] = -1.0 / hy
                                    A[K, idx_P(i, j - 1, n)] = 1.0 / hy
                                    A[K, K] = -2.0 * C * (1.0/hx**2 + 1.0/hy**2)
                                    A[K, idx_V(i + 1, j, n, m) + Np + Nu] = C / hx**2
                                    A[K, idx_V(i - 1, j, n, m) + Np + Nu] = C / hx**2
                                    A[K, idx_V(i, j + 1, n, m) + Np + Nu] = C / hy**2
                                    A[K, idx_V(i, j - 1, n, m) + Np + Nu] = C / hy**2
                                    
                        return A.tocsr(), B

                    # Boucle adaptation géométrie
                    iteration = 0
                    changement = True
                    R_history = [R.copy()]
                    
                    while iteration < max_iterations and changement:
                        A, B = matrixes(n, m, C, hx, hy, P_entree, P_sortie, R, l, Np, Nu, Nv)
                        
                        try:
                            S = spsolve(A, B)
                        except Exception as e:
                            break
                        
                        offset_P = 0
                        offset_U = Np
                        offset_V = Np + Nu
                        P_vecteur = S[offset_P:offset_U]
                        P_grille = P_vecteur.reshape((m, n))
                        
                        # Extraction pression paroi
                        pression_au_mur = np.zeros(n)
                        y_centre_phys = l / 2.0
                        for i in range(n): 
                            R_local_p = (R[i] + R[i+1]) / 2.0
                            y_mur_sup_phys = y_centre_phys + R_local_p
                            j_mur_idx = int((y_mur_sup_phys / hy) - 0.5 - 1e-6)
                            j_mur_idx = min(max(j_mur_idx, 0), m - 1)
                            pression_au_mur[i] = P_grille[j_mur_idx, i]

                        R_ancien = R.copy()
                        changement = False
                        max_change = 0
                        
                        # Mise à jour géométrie
                        for i in range(n + 1):
                            if i == 0:
                                pression_interieure = pression_au_mur[0]
                            elif i == n:
                                pression_interieure = pression_au_mur[n-1]
                            else:
                                pression_interieure = (pression_au_mur[i-1] + pression_au_mur[i]) / 2.0
                            
                            R_new = D_tube(pression_interieure, P0, P1_const, P2_const, 
                                          n1_const, n2_const, alpha0_val, 2*R0) / 2.0
                            
                            # Relaxation pour convergence
                            R_new = R_ancien[i] + (R_new - R_ancien[i]) * 0.5
                            
                            change = abs(R_new - R_ancien[i])
                            if change > max_change:
                                max_change = change
                            if change > tolerance:
                                changement = True

                            R[i] = R_new

                        R_history.append(R.copy())
                        iteration += 1

                    # Extraction résultats finaux
                    P_vecteur = S[offset_P:offset_U]
                    U_vecteur = S[offset_U:offset_V]
                    V_vecteur = S[offset_V:]
                    
                    P_grille_stag = P_vecteur.reshape((m, n))
                    U_grille_stag = U_vecteur.reshape((m, n + 1))
                    V_grille_stag = V_vecteur.reshape((m + 1, n))
                    
                    P_grille = P_grille_stag
                    Ux_grille = (U_grille_stag[:, :-1] + U_grille_stag[:, 1:]) / 2.0
                    Uy_grille = (V_grille_stag[:-1, :] + V_grille_stag[1:, :]) / 2.0
                    
                    resultats[gen] = {
                        'generation': gen,
                        'rayon_final': R,
                        'rayon_repos': R_repos,
                        'rayon_max': R0,
                        'pression_entree': P_entree,
                        'pression_sortie': P_sortie,
                        'P_grille': P_grille,
                        'Ux_grille': Ux_grille,
                        'Uy_grille': Uy_grille,
                        'n': n,
                        'm': m,
                        'L': L,
                        'l': l,
                        'hx': hx,
                        'hy': hy,
                        'convergence_iterations': iteration
                    }
                    
                except Exception as e:
                    print(f"Erreur génération {gen}: {e}")
                    continue

            # Analyse comparative
            self.after(0, self.afficher_graphique_toutes_generations, resultats, cas)
            
        except Exception as e:
            self.after(0, self.afficher_erreur, f"Erreur toutes générations: {str(e)}")

    def mettre_a_jour_progression(self, gen_actuel, total_generations):
        progression = (gen_actuel + 1) / total_generations * 100
        self.gen_progress['value'] = progression
        self.gen_progress_label['text'] = f"{gen_actuel + 1}/{total_generations} générations"
        self.status_var.set(f"Simulation génération {gen_actuel + 1}/{total_generations}")

    def afficher_graphique_toutes_generations(self, resultats, cas):
        for widget in self.gen_figure_frame.winfo_children():
            widget.destroy()
            
        # Notebook pour organiser les graphiques
        notebook = ttk.Notebook(self.gen_figure_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Préparation des données pour les graphiques comparatifs
        generations = []
        rayons_finaux_moyens = []
        rayons_repos = []
        variations_relatives = []
        pressions_entree = []
        pressions_sortie = []

        for gen in sorted(resultats.keys()):
            data = resultats[gen]
            rayon_final_moyen = np.mean(data['rayon_final'])
            rayon_repos = data['rayon_repos']
            variation = (rayon_final_moyen - rayon_repos) / rayon_repos * 100
            
            generations.append(gen)
            rayons_finaux_moyens.append(rayon_final_moyen * 1000)  # en mm
            rayons_repos.append(rayon_repos * 1000)  # en mm
            variations_relatives.append(variation)
            pressions_entree.append(data['pression_entree'])
            pressions_sortie.append(data['pression_sortie'])

        # Graphique 1: Comparaison des rayons
        frame1 = ttk.Frame(notebook)
        notebook.add(frame1, text="Comparaison Rayons")
        
        fig1 = Figure(figsize=(12, 8))
        ax1 = fig1.add_subplot(111)
        
        ax1.plot(generations, rayons_repos, 'go-', linewidth=2, markersize=6, label='Rayon au repos')
        ax1.plot(generations, rayons_finaux_moyens, 'bo-', linewidth=2, markersize=6, label='Rayon final')
        ax1.set_xlabel('Génération')
        ax1.set_ylabel('Rayon (mm)')
        ax1.set_title(f'Évolution des rayons bronchiques - {cas}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(generations)
        
        # Zones de compression/dilatation
        ax1.axvspan(0, 4, alpha=0.2, color='red', label='Compression (générations centrales)')
        ax1.axvspan(5, 16, alpha=0.2, color='green', label='Dilatation (générations périphériques)')
        ax1.legend()
        
        canvas1 = FigureCanvasTkAgg(fig1, frame1)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Graphique 2: Variation relative
        frame2 = ttk.Frame(notebook)
        notebook.add(frame2, text="Variation Relative")
        
        fig2 = Figure(figsize=(12, 8))
        ax2 = fig2.add_subplot(111)
        
        bars = ax2.bar(generations, variations_relatives, 
                      color=['red' if v < 0 else 'green' for v in variations_relatives])
        ax2.set_xlabel('Génération')
        ax2.set_ylabel('Variation relative (%)')
        ax2.set_title(f'Variation du rayon pendant {cas}')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(generations)
        
        # Ajout des valeurs sur les barres
        for i, v in enumerate(variations_relatives):
            ax2.text(generations[i], v + (1 if v >= 0 else -3), f'{v:.1f}%', 
                    ha='center', va='bottom' if v >= 0 else 'top', fontweight='bold')
        
        canvas2 = FigureCanvasTkAgg(fig2, frame2)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Graphique 3: Pressions
        frame3 = ttk.Frame(notebook)
        notebook.add(frame3, text="Pressions")
        
        fig3 = Figure(figsize=(12, 8))
        ax3 = fig3.add_subplot(111)
        
        ax3.plot(generations, pressions_entree, 's-', linewidth=2, markersize=6, label='Pression entrée')
        ax3.plot(generations, pressions_sortie, '^-', linewidth=2, markersize=6, label='Pression sortie')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Génération')
        ax3.set_ylabel('Pression transmurale (Pa)')
        ax3.set_title(f'Pressions appliquées - {cas}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(generations)
        
        # Zones de compression/dilatation
        ax3.axvspan(0, 4, alpha=0.2, color='red')
        ax3.axvspan(5, 16, alpha=0.2, color='green')
        
        canvas3 = FigureCanvasTkAgg(fig3, frame3)
        canvas3.draw()
        canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Graphique 4: Paradoxe bronchique
        frame4 = ttk.Frame(notebook)
        notebook.add(frame4, text="Paradoxe Bronchique")
        
        fig4 = Figure(figsize=(12, 8))
        ax4 = fig4.add_subplot(111)
        
        # Séparation des générations centrales et périphériques
        gen_centrales = [g for g in generations if g <= 4]
        gen_peripheriques = [g for g in generations if g >= 5]
        
        var_centrales = [v for g, v in zip(generations, variations_relatives) if g <= 4]
        var_peripheriques = [v for g, v in zip(generations, variations_relatives) if g >= 5]
        
        ax4.bar(['Centrales\n(0-4)', 'Périphériques\n(5-16)'], 
                [np.mean(var_centrales), np.mean(var_peripheriques)],
                color=['red', 'green'], alpha=0.7)
        ax4.set_ylabel('Variation moyenne du rayon (%)')
        ax4.set_title('Paradoxe de l\'expiration forcée\nCompression centrale vs Dilatation périphérique')
        ax4.grid(True, alpha=0.3)
        
        # Ajout des valeurs
        ax4.text(0, np.mean(var_centrales) + (1 if np.mean(var_centrales) >= 0 else -3), 
                f'{np.mean(var_centrales):.1f}%', ha='center', va='bottom' if np.mean(var_centrales) >= 0 else 'top', 
                fontweight='bold', fontsize=12)
        ax4.text(1, np.mean(var_peripheriques) + (1 if np.mean(var_peripheriques) >= 0 else -3), 
                f'{np.mean(var_peripheriques):.1f}%', ha='center', va='bottom' if np.mean(var_peripheriques) >= 0 else 'top', 
                fontweight='bold', fontsize=12)
        
        canvas4 = FigureCanvasTkAgg(fig4, frame4)
        canvas4.draw()
        canvas4.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.status_var.set(f"Simulation de toutes les générations terminée ({len(resultats)}/17 générations)")

    # Méthodes utilitaires
    def afficher_erreur(self, message):
        messagebox.showerror("Erreur", message)
        self.status_var.set("Erreur lors de la simulation")
        
    def afficher_message(self, message):
        messagebox.showinfo("Information", message)
        self.status_var.set("Prêt")

if __name__ == "__main__":
    app = SimulateurBronchique()
    app.mainloop()













input("fin")