import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from rec import RecommendationSystem

DATA_FILE = "high_popularity_spotify_data (1).csv"


class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)

        canvas = tk.Canvas(
            self, borderwidth=0, background="#f0f0f0", highlightthickness=0
        )
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas, padding="10")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")


class RecommenderApp:
    def __init__(self, master):
        self.master = master
        master.title("VibeMatch: Smart Music Recommender")
        master.geometry("1100x850")
        master.configure(bg="#f0f0f0")

        self.style = self._setup_styles()

        try:
            self.model = RecommendationSystem(DATA_FILE)
        except FileNotFoundError as e:
            messagebox.showerror("Fatal Error", str(e))
            master.destroy()
            return

        self.k_value = tk.IntVar(value=10)
        self.search_term = tk.StringVar()
        self.selected_track_id = tk.StringVar(
            value=self.model.df.index[0] if not self.model.df.empty else ""
        )

        self.current_listener_name = tk.StringVar()
        self.selected_user_id = tk.StringVar()
        self.test_user_var = tk.StringVar()

        self.found_tracks = []

        self._setup_ui()
        self._initial_load()
        self._refresh_listener_lists()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#f0f0f0")
        style.configure("TLabel", background="#f0f0f0", font=("Arial", 10))
        style.configure(
            "Title.TLabel", font=("Arial", 18, "bold"), foreground="#1DB954"
        )
        style.configure("Header.TLabel", font=("Arial", 12, "bold"))
        style.configure(
            "TButton",
            font=("Arial", 10, "bold"),
            background="#1DB954",
            foreground="black",
        )
        style.map("TButton", background=[("active", "#1ED760")])
        style.configure("TNotebook.Tab", font=("Arial", 10, "bold"))
        return style

    def _setup_ui(self):
        title_frame = ttk.Frame(self.master, padding="10")
        title_frame.pack(fill="x")

        ttk.Label(
            title_frame,
            text="VibeMatch: Smart Music Recommender",
            style="Title.TLabel",
        ).pack(side="top", anchor="center", pady=(5, 2))

        ttk.Label(
            title_frame,
            text="Compare song-based and people-based recommendations with live listeners",
            font=("Arial", 10),
            background="#f0f0f0",
        ).pack(side="top", anchor="center", pady=(0, 5))

        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(pady=10, padx=10, expand=True, fill="both")

        recommender_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(recommender_frame, text="Discover Music")
        self._setup_recommendation_tab(recommender_frame)

        evaluation_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(evaluation_frame, text="Quality & Comparison")
        self._setup_evaluation_tab(evaluation_frame)

    def _setup_recommendation_tab(self, parent):
        main_paned_window = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        main_paned_window.pack(fill="both", expand=True)

        # LEFT side: controls in scrollable frame
        control_scroll = ScrollableFrame(main_paned_window)
        control_frame = control_scroll.scrollable_frame
        main_paned_window.add(control_scroll, weight=0)

        ttk.Label(
            control_frame,
            text="Listener setup",
            style="Header.TLabel",
        ).pack(pady=(5, 5), anchor="w")

        self.name_label = ttk.Label(
            control_frame, text="Who are we saving this like for?"
        )
        self.name_label.pack(anchor="w")
        self.listener_name_entry = ttk.Entry(
            control_frame, textvariable=self.current_listener_name, width=25
        )
        self.listener_name_entry.pack(anchor="w", pady=(0, 5))

        self.add_button = ttk.Button(
            control_frame,
            text="Save selected song as liked",
            command=self.add_song_to_listener,
            style="TButton",
        )
        self.add_button.pack(anchor="w", pady=(0, 10), fill="x")

        ttk.Label(
            control_frame,
            text="Recommendation mode",
            style="Header.TLabel",
        ).pack(pady=(10, 5), anchor="w")

        self.model_choice = tk.StringVar(value="Song-Based")

        ttk.Radiobutton(
            control_frame,
            text="Song-based: similar audio to a track",
            variable=self.model_choice,
            value="Song-Based",
            command=self._toggle_input_mode,
        ).pack(anchor="w")

        ttk.Radiobutton(
            control_frame,
            text="People-based: similar listeners like you",
            variable=self.model_choice,
            value="People-Based",
            command=self._toggle_input_mode,
        ).pack(anchor="w")

        ttk.Label(
            control_frame,
            text="\nHow many songs should we suggest?",
            style="Header.TLabel",
        ).pack(pady=(10, 5), anchor="w")

        self.k_entry = ttk.Entry(control_frame, textvariable=self.k_value, width=10)
        self.k_entry.pack(anchor="w", pady=(0, 10))

        self.cbf_input_frame = ttk.Frame(control_frame)
        self.cbf_input_frame.pack(fill="x")

        ttk.Label(
            self.cbf_input_frame,
            text="Pick a starting song",
            style="Header.TLabel",
        ).pack(pady=(10, 5), anchor="w")

        ttk.Label(self.cbf_input_frame, text="Search by title or artist:").pack(
            anchor="w"
        )
        self.search_entry = ttk.Entry(
            self.cbf_input_frame, textvariable=self.search_term, width=40
        )
        self.search_entry.pack(fill="x", pady=2)
        self.search_entry.bind("<KeyRelease>", self._search_tracks)

        ttk.Label(self.cbf_input_frame, text="Search results:").pack(
            anchor="w", pady=(5, 0)
        )
        self.track_listbox = tk.Listbox(
            self.cbf_input_frame,
            height=10,
            width=40,
            exportselection=0,
        )
        self.track_listbox.pack(fill="both", expand=True, pady=(0, 10))
        self.track_listbox.bind("<<ListboxSelect>>", self._on_track_select)

        self.cf_input_frame = ttk.Frame(control_frame)

        ttk.Label(
            self.cf_input_frame,
            text="Who are we recommending for?",
            style="Header.TLabel",
        ).pack(pady=(10, 5), anchor="w")

        self.user_selector = ttk.Combobox(
            self.cf_input_frame,
            textvariable=self.selected_user_id,
            values=[],
            state="readonly",
            width=30,
        )
        self.user_selector.pack(fill="x", pady=(0, 15))

        ttk.Button(
            self.cf_input_frame,
            text="Manage this listener's liked songs",
            command=self.open_liked_songs_manager,
            style="TButton",
        ).pack(fill="x", pady=(0, 10))

        self.view_likes_button = ttk.Button(
            self.cf_input_frame,
            text="View this listener's liked songs",
            command=self.show_liked_songs,
            style="TButton",
        )
        self.view_likes_button.pack(fill="x", pady=(0, 10))

        ttk.Button(
            control_frame,
            text="GET SUGGESTIONS",
            command=self.run_recommendation,
            style="TButton",
        ).pack(pady=(10, 10), fill="x")

        # RIGHT side: results
        results_frame = ttk.Frame(main_paned_window, padding="10")
        main_paned_window.add(results_frame, weight=1)

        ttk.Label(
            results_frame,
            text="Suggestions",
            style="Header.TLabel",
        ).pack(pady=(5, 10), anchor="w")

        self.results_text = tk.Text(
            results_frame,
            height=20,
            width=70,
            state=tk.DISABLED,
            wrap=tk.WORD,
            font=("Courier", 10),
        )
        self.results_text.pack(fill="both", expand=True)

        self.details_label = ttk.Label(
            results_frame,
            text="",
            font=("Arial", 10),
            justify="left",
        )
        self.details_label.pack(fill="x", pady=(10, 0), anchor="w")

        # For the "liked songs manager" window
        self.likes_manager_window = None
        self.likes_manager_listbox = None
        self.likes_manager_tracks = []
        self.likes_manager_listener = None

    def open_liked_songs_manager(self):
        """Open a small window showing this listener's liked songs and allow removals."""
        listener = self.user_selector.get().strip()
        if not listener:
            messagebox.showwarning(
                "No listener selected", "Please choose a listener first."
            )
            return

        df, error = self.model.get_liked_tracks_for_listener(listener)
        if error:
            messagebox.showerror("MongoDB error", error)
            return

        if df is None or df.empty:
            messagebox.showinfo(
                "No liked songs",
                f"{listener} has no liked songs stored yet.",
            )
            return

        if self.likes_manager_window is not None and self.likes_manager_window.winfo_exists():
            self.likes_manager_window.destroy()

        win = tk.Toplevel(self.master)
        win.title(f"{listener}'s liked songs")
        win.geometry("550x400")
        win.configure(bg="#f0f0f0")
        self.likes_manager_window = win
        self.likes_manager_listener = listener

        ttk.Label(
            win,
            text=f"Liked songs for {listener}",
            style="Header.TLabel",
        ).pack(pady=5, anchor="center")

        listbox = tk.Listbox(
            win,
            height=15,
            width=70,
            exportselection=False,
        )
        listbox.pack(fill="both", expand=True, padx=10, pady=5)
        self.likes_manager_listbox = listbox

        self.likes_manager_tracks = list(df.index)

        for tid, row in df.iterrows():
            display = f"{row['Track']} – {row['Artist']} ({row['playlist_genre']})"
            listbox.insert(tk.END, display)

        ttk.Button(
            win,
            text="Remove selected song from likes",
            command=self.remove_selected_like,
            style="TButton",
        ).pack(pady=10)

    def remove_selected_like(self):
        """Remove the selected song from the current listener's liked songs."""
        if not self.likes_manager_window or not self.likes_manager_listbox:
            return

        selection = self.likes_manager_listbox.curselection()
        if not selection:
            messagebox.showwarning(
                "No song selected",
                "Please select a song to remove from this listener's likes.",
            )
            return

        idx = selection[0]
        track_id = self.likes_manager_tracks[idx]
        listener = self.likes_manager_listener

        confirm = messagebox.askyesno(
            "Confirm removal",
            f"Remove this song from {listener}'s liked songs?",
        )
        if not confirm:
            return

        error = self.model.remove_like_for_listener(listener, track_id)
        if error:
            messagebox.showerror("MongoDB error", error)
            return

        self.likes_manager_listbox.delete(idx)
        del self.likes_manager_tracks[idx]

        matrix_error = self.model.build_user_item_matrix_from_mongo()
        if matrix_error:
            messagebox.showwarning("Matrix warning", matrix_error)

        self._refresh_listener_lists()

        if self.likes_manager_listbox.size() == 0:
            messagebox.showinfo(
                "No more liked songs",
                f"{listener} has no more liked songs.",
            )
            self.likes_manager_window.destroy()
            self.likes_manager_window = None

    def _setup_evaluation_tab(self, parent):
        eval_paned_window = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        eval_paned_window.pack(fill="both", expand=True)

        # LEFT: controls + text (with its own scrollbar)
        metrics_frame = ttk.Frame(eval_paned_window, width=320, padding="10")
        eval_paned_window.add(metrics_frame, weight=0)

        ttk.Label(
            metrics_frame,
            text="Evaluate recommendation quality",
            style="Header.TLabel",
        ).pack(pady=(10, 5), anchor="w")

        ttk.Label(
            metrics_frame,
            text="Which listener do we test the engines on?",
        ).pack(anchor="w")
        self.test_user_selector = ttk.Combobox(
            metrics_frame,
            textvariable=self.test_user_var,
            values=[],
            state="readonly",
            width=30,
        )
        self.test_user_selector.pack(fill="x", pady=(0, 15))

        ttk.Button(
            metrics_frame,
            text="Run quality comparison",
            command=self.run_ab_test,
            style="TButton",
        ).pack(pady=(10, 20), fill="x")

        ttk.Label(
            metrics_frame,
            text="Results:",
            style="Header.TLabel",
        ).pack(anchor="w")

        # Text + scrollbar container
        text_frame = ttk.Frame(metrics_frame)
        text_frame.pack(fill="both", expand=True, pady=(5, 10))

        text_scroll = ttk.Scrollbar(text_frame, orient="vertical")
        self.eval_text = tk.Text(
            text_frame,
            height=15,
            width=40,
            state=tk.DISABLED,
            wrap=tk.WORD,
            font=("Courier", 10),
            yscrollcommand=text_scroll.set,
        )
        text_scroll.config(command=self.eval_text.yview)
        self.eval_text.pack(side="left", fill="both", expand=True)
        text_scroll.pack(side="right", fill="y")

        # RIGHT: plots inside a scrollable frame
        plot_scroll = ScrollableFrame(eval_paned_window)
        plot_container = plot_scroll.scrollable_frame
        eval_paned_window.add(plot_scroll, weight=1)

        ttk.Label(
            plot_container,
            text="Song-based vs people-based score",
            style="Header.TLabel",
        ).pack(pady=(5, 10), anchor="w")

        # Bar chart: AP@10 scores
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.ax.set_title("Recommendation quality (AP@10)", fontsize=12)
        self.ax.set_ylim(0, 1.0)
        self.ax.set_ylabel("Match score")
        self.ax.set_xticks([0, 1])
        self.ax.set_xticklabels(["Song-based", "People-based"])
        self.ax.bar([0, 1], [0, 0])

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_container)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)

        # Second plot: listener's favourite genres (blue)
        self.fig_genres, self.ax_genres = plt.subplots(figsize=(6, 3))
        self.ax_genres.set_title("Listener's top liked genres", fontsize=12)
        self.ax_genres.set_ylabel("Count")

        self.canvas_genres = FigureCanvasTkAgg(self.fig_genres, master=plot_container)
        self.canvas_genres_widget = self.canvas_genres.get_tk_widget()
        self.canvas_genres_widget.pack(fill="both", expand=True, pady=(10, 0))

    def _initial_load(self):
        self._search_tracks(None)
        if self.track_listbox.size() > 0:
            self.track_listbox.select_set(0)
            self.track_listbox.event_generate("<<ListboxSelect>>")
        self._toggle_input_mode()

    def _toggle_input_mode(self):
        mode = self.model_choice.get()

        if mode == "Song-Based":
            # Show TRACK INPUT (search bar) + name + like button
            if not self.name_label.winfo_ismapped():
                self.name_label.pack(anchor="w")
                self.listener_name_entry.pack(anchor="w", pady=(0, 5))
            if not self.add_button.winfo_ismapped():
                self.add_button.pack(anchor="w", pady=(0, 10), fill="x")

            self.cf_input_frame.pack_forget()
            self.cbf_input_frame.pack(fill="x")

        elif mode == "People-Based":
            # Hide like button + name entry, show listener selector
            if self.add_button.winfo_ismapped():
                self.add_button.pack_forget()
            if self.name_label.winfo_ismapped():
                self.name_label.pack_forget()
                self.listener_name_entry.pack_forget()

            self.cbf_input_frame.pack_forget()
            self.cf_input_frame.pack(fill="x")

    def _search_tracks(self, event):
        search = self.search_term.get().lower().strip()
        self.track_listbox.delete(0, tk.END)
        self.found_tracks = []

        if len(search) <= 1:
            filtered_df = self.model.df.head(50)
        else:
            filtered_df = self.model.df[
                self.model.df["Track"].str.lower().str.contains(search, na=False)
                | self.model.df["Artist"].str.lower().str.contains(search, na=False)
            ].head(100)

        for index, row in filtered_df.iterrows():
            display_text = f"{row['Track']} by {row['Artist']}"
            self.track_listbox.insert(tk.END, display_text)
            self.found_tracks.append(index)

    def _on_track_select(self, event):
        selected_indices = self.track_listbox.curselection()
        if selected_indices:
            idx = selected_indices[0]
            if 0 <= idx < len(self.found_tracks):
                track_id = self.found_tracks[idx]
                self.selected_track_id.set(track_id)

    def _update_results_text(self, title: str, recommendations, engine_label: str):
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)

        header = f"{title}\n"
        header += f"Engine: {engine_label}\n\n"

        if recommendations is None or recommendations.empty:
            self.results_text.insert(tk.END, f"{header}No suggestions found.")
        else:
            formatted_table = []
            formatted_table.append(
                f"{'#':<3}{'Track Name':<40}{'Artist':<30}{'Genre':<15}"
            )
            formatted_table.append("-" * 90)

            for i, (_, row) in enumerate(recommendations.iterrows()):
                formatted_table.append(
                    f"{i + 1:<3}"
                    f"{row['Track'][:38]:<40}"
                    f"{row['Artist'][:28]:<30}"
                    f"{row['playlist_genre'][:13]:<15}"
                )

            self.results_text.insert(tk.END, header + "\n".join(formatted_table))

        self.results_text.config(state=tk.DISABLED)

    def _update_details_panel_for_song_based(self, seed_id: str):
        if not seed_id or seed_id not in self.model.df.index:
            self.details_label.config(text="")
            return

        track_row = self.model.df.loc[seed_id]
        details = (
            "Song-based mode\n"
            "We find tracks with similar audio characteristics.\n\n"
            "Starting song:\n"
            f"• Title : {track_row['Track']}\n"
            f"• Artist: {track_row['Artist']}\n"
            f"• Genre : {track_row['playlist_genre']}\n"
        )
        self.details_label.config(text=details)

    def _update_details_panel_for_people_based(self, user_id: str):
        if (
            self.model.user_item_matrix is None
            or user_id not in self.model.user_item_matrix.index
        ):
            self.details_label.config(text="")
            return

        user_likes = self.model.user_item_matrix.loc[user_id]
        n_likes = int((user_likes > 0).sum())
        details = (
            "People-based mode\n"
            "We look for listeners who like similar songs,\n"
            "then suggest tracks they also enjoy.\n\n"
            f"Personalized for: {user_id}\n"
            f"• Liked songs stored: {n_likes}\n"
        )
        self.details_label.config(text=details)

    def show_liked_songs(self):
        user = self.user_selector.get().strip()
        if not user:
            messagebox.showwarning(
                "Missing listener", "Please choose a listener from the list first."
            )
            return

        liked_df, error = self.model.get_liked_tracks_for_listener(user)

        if error:
            messagebox.showinfo("Liked songs", error)
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"{error}")
            self.results_text.config(state=tk.DISABLED)
            return

        self._update_results_text(
            f"{user}'s liked songs",
            liked_df,
            "Saved likes",
        )
        self._update_details_panel_for_people_based(user)

    def add_song_to_listener(self):
        name = self.current_listener_name.get().strip()
        if not name:
            messagebox.showwarning(
                "Missing name", "Please enter the listener's name first."
            )
            return

        track_id = self.selected_track_id.get()
        if not track_id:
            messagebox.showwarning(
                "Missing song", "Please select a song from the list."
            )
            return

        error = self.model.add_like_for_listener(name, track_id)
        if error:
            messagebox.showerror("Database error", error)
            return

        self._refresh_listener_lists()
        messagebox.showinfo(
            "Profile updated", f"{name}'s liked songs were updated in MongoDB."
        )

    def _refresh_listener_lists(self):
        names = self.model.get_listener_names()
        self.user_selector["values"] = names
        self.test_user_selector["values"] = names

        if names:
            if self.selected_user_id.get() not in names:
                self.selected_user_id.set(names[0])
            if self.test_user_var.get() not in names:
                self.test_user_var.set(names[0])

    def run_recommendation(self):
        try:
            k = int(self.k_value.get())
            if k <= 0 or k > 50:
                raise ValueError("K must be between 1 and 50.")
        except ValueError as e:
            messagebox.showwarning("Input Error", f"Invalid number of songs: {e}")
            return

        mode = self.model_choice.get()
        recommendations = None
        error = None

        if mode == "Song-Based":
            seed_id = self.selected_track_id.get()
            if not seed_id:
                messagebox.showwarning("Missing song", "Please pick a starting song.")
                return

            track_name = self.model.df.loc[seed_id]["Track"]
            recommendations, error = self.model.get_content_based_recommendations(
                seed_id, k
            )
            self._update_results_text(
                f"Songs similar to: {track_name}",
                recommendations,
                "Song-based (audio similarity)",
            )
            self._update_details_panel_for_song_based(seed_id)

        elif mode == "People-Based":
            if not self.model.get_listener_names():
                messagebox.showwarning(
                    "No listeners", "Please add at least one listener with liked songs."
                )
                return

            build_error = self.model.build_user_item_matrix_from_mongo()
            if build_error:
                messagebox.showwarning("Listener data problem", build_error)
                return

            user_id = self.user_selector.get().strip()
            if not user_id:
                messagebox.showwarning(
                    "Missing listener", "Please choose a listener from the list."
                )
                return

            recommendations, error = self.model.simulate_collaborative_filtering(
                user_id, k
            )
            self._update_results_text(
                f"Personalized mix for {user_id}",
                recommendations,
                "People-based (similar listeners)",
            )
            self._update_details_panel_for_people_based(user_id)

        if error:
            messagebox.showerror("Suggestions Error", error)
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Error: {error}")
            self.results_text.config(state=tk.DISABLED)

    def run_ab_test(self):
        if not self.model.get_listener_names():
            messagebox.showwarning(
                "No listeners", "Please add at least one listener with liked songs."
            )
            return

        build_error = self.model.build_user_item_matrix_from_mongo()
        if build_error:
            messagebox.showwarning("Listener data problem", build_error)
            return

        test_user = self.test_user_var.get().strip()
        if not test_user:
            messagebox.showwarning(
                "Missing listener", "Please pick a listener for the comparison."
            )
            return

        user_likes = self.model.user_item_matrix.loc[test_user][
            self.model.user_item_matrix.loc[test_user] > 0
        ]
        if user_likes.empty:
            messagebox.showwarning(
                "Evaluation Error",
                f"{test_user} has no liked songs.",
            )
            return

        k = 10
        cbf_seed_track_id = user_likes.index[0]

        song_based_score, song_error = self.model.evaluate_model(
            self.model.get_content_based_recommendations,
            test_user_id=test_user,
            test_track_id=cbf_seed_track_id,
            k=k,
        )

        people_based_score, people_error = self.model.evaluate_model(
            self.model.simulate_collaborative_filtering,
            test_user_id=test_user,
            k=k,
        )

        self.eval_text.config(state=tk.NORMAL)
        self.eval_text.delete(1.0, tk.END)
        self.eval_text.insert(
            tk.END,
            f"Listener: {test_user}\n\n",
        )
        self.eval_text.insert(
            tk.END,
            "We compare two engines:\n"
            "A = Song-based (similar audio features)\n"
            "B = People-based (learn from similar listeners)\n\n"
        )
        self.eval_text.insert(tk.END, "Score range: 0 (weak) to 1 (strong)\n\n")

        if song_error:
            self.eval_text.insert(tk.END, f"A: Song-based score: N/A ({song_error})\n")
            song_based_score = 0.0
        else:
            self.eval_text.insert(
                tk.END, f"A: Song-based score: {song_based_score:.4f}\n"
            )

        if people_error:
            self.eval_text.insert(
                tk.END, f"B: People-based score: N/A ({people_error})\n"
            )
            people_based_score = 0.0
        else:
            self.eval_text.insert(
                tk.END, f"B: People-based score: {people_based_score:.4f}\n"
            )

        self.eval_text.insert(tk.END, "\nSummary:\n")
        if song_based_score > people_based_score:
            self.eval_text.insert(
                tk.END,
                "For this listener, the song-based engine matches their taste slightly better.\n",
            )
        elif people_based_score > song_based_score:
            self.eval_text.insert(
                tk.END,
                "For this listener, the people-based engine matches their taste slightly better.\n",
            )
        else:
            self.eval_text.insert(
                tk.END,
                "Both engines perform about the same for this listener.\n",
            )

        self.eval_text.config(state=tk.DISABLED)

        self.ax.clear()
        scores = [song_based_score, people_based_score]
        colors = [
            "#1DB954" if scores[0] >= scores[1] else "#90EE90",
            "#1E90FF" if scores[1] >= scores[0] else "#ADD8E6",
        ]

        self.ax.bar(["Song-based", "People-based"], scores, color=colors)
        self.ax.set_title("Recommendation quality (AP@10)", fontsize=12)
        self.ax.set_ylim(0, 1.0)
        self.ax.set_ylabel("Match score")

        for i, score in enumerate(scores):
            self.ax.text(i, score + 0.02, f"{score:.4f}", ha="center", fontsize=10)

        self.fig.tight_layout()
        self.canvas.draw()

        self._update_genre_plot(test_user)

    def _update_genre_plot(self, user_id: str):
        if (
            self.model.user_item_matrix is None
            or user_id not in self.model.user_item_matrix.index
        ):
            self.ax_genres.clear()
            self.ax_genres.set_title("Listener's top liked genres", fontsize=12)
            self.ax_genres.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=self.ax_genres.transAxes,
            )
            self.fig_genres.tight_layout()
            self.canvas_genres.draw()
            return

        user_likes = self.model.user_item_matrix.loc[user_id]
        liked_ids = user_likes[user_likes > 0].index.tolist()

        self.ax_genres.clear()
        self.ax_genres.set_title("Listener's top liked genres", fontsize=12)

        if not liked_ids:
            self.ax_genres.text(
                0.5,
                0.5,
                "This listener has no liked songs yet.",
                ha="center",
                va="center",
                transform=self.ax_genres.transAxes,
            )
        else:
            genre_counts = (
                self.model.df.loc[liked_ids, "playlist_genre"]
                .value_counts()
                .head(8)
            )
            self.ax_genres.bar(genre_counts.index, genre_counts.values)
            self.ax_genres.set_ylabel("Count")
            self.ax_genres.set_xticklabels(
                genre_counts.index, rotation=30, ha="right"
            )

        self.fig_genres.tight_layout()
        self.canvas_genres.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = RecommenderApp(root)
    root.mainloop()
