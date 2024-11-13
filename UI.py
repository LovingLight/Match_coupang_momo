import tkinter as tk
import os
import pandas as pd
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import messagebox
import matplotlib.pyplot as plt


class MatchingResultsApp:
    def __init__(self, root, queries_file, folder_v1, folder_example):
        self.root = root
        self.queries_file = queries_file
        self.folder_v1 = folder_v1
        self.folder_example = folder_example

        # 設定視窗大小
        self.root.geometry("1500x800")

        self.queries = self.load_queries()

        self.selected_query = tk.StringVar(self.root)
        self.selected_query.set(self.queries[0])  # 默認選擇第一個搜尋詞

        self.create_widgets()

    def create_widgets(self):
        """創建 UI 控制項"""
        # 下拉選單選擇搜尋詞，使用 ttk.Combobox 顯示可滾動的選單
        query_dropdown = ttk.Combobox(
            self.root, textvariable=self.selected_query, values=self.queries, state="readonly")
        query_dropdown.pack(pady=10, fill='x')  # 設定為 x 軸填充，這樣看起來更整齊

        # 比較結果按鈕
        compare_button = tk.Button(
            self.root, text="Compare", command=lambda: self.compare_results(self.selected_query.get()))
        compare_button.pack(pady=10)

        # 顯示匹配結果的框架
        self.results_frame = tk.Frame(self.root)
        self.results_frame.pack(pady=10)

        # 顯示分布圖的按鈕
        plot_button = tk.Button(
            self.root, text="Show Similarity Distribution", command=self.plot_similarity_distribution)
        plot_button.pack(side="bottom", pady=10)

    def load_queries(self):
        """載入搜尋詞列表"""
        try:
            with open(self.queries_file, 'r', encoding='utf-8') as file:
                queries = file.readlines()
            return [query.strip() for query in queries]
        except FileNotFoundError:
            print(f"Error: {self.queries_file} not found.")
            return []

    def compare_results(self, query):
        """讀取兩個資料夾中的 CSV 檔案，並比較匹配結果"""
        # 讀取 v1 資料夾和 Example 資料夾中的 CSV 檔案
        file_v1 = os.path.join(self.folder_v1, f"{query}.csv")
        file_example = os.path.join(self.folder_example, f"{query}.csv")

        if os.path.exists(file_v1) and os.path.exists(file_example):
            # 讀取 CSV 檔案
            df_v1 = pd.read_csv(file_v1)
            df_example = pd.read_csv(file_example)

            # 顯示 CSV 檔案的欄位名稱，幫助檢查
            print("Columns in v1 file:", df_v1.columns)
            print("Columns in example file:", df_example.columns)

            max_items = 30

            self.display_results(df_v1[:max_items], df_example[:max_items])

    def display_results(self, df_our, df_example):
        """顯示比較結果"""
        self.clear_results()

        df_our_sorted, df_example_sorted = self.prepare_data(
            df_our, df_example)

        self.create_results_section(
            df_our_sorted, title="Our Matching Results", column=0)
        self.create_results_section(
            df_example_sorted, title="Example Matching Results", column=1)

    def clear_results(self):
        """清除舊的結果顯示"""
        for widget in self.results_frame.winfo_children():
            widget.destroy()

    def prepare_data(self, df_our, df_example):
        """合併和標記匹配資料，並排序將匹配行移到最上層"""
        df_merged = pd.merge(df_our, df_example, on=[
                             'MOMO Title', 'Product Title'], suffixes=('_our', '_example'))
        df_merged['highlight'] = True

        df_our['highlight'] = df_our.apply(
            lambda row: row['MOMO Title'] in df_merged['MOMO Title'].values and row['Product Title'] in df_merged['Product Title'].values, axis=1)
        df_example['highlight'] = df_example.apply(
            lambda row: row['MOMO Title'] in df_merged['MOMO Title'].values and row['Product Title'] in df_merged['Product Title'].values, axis=1)

        # 排除 Similarity 為 NaN 的行
        df_our = df_our.dropna(subset=['Similarity'])
        df_example = df_example.dropna(subset=['Similarity'])

        df_our_sorted = df_our.sort_values(
            by='highlight', ascending=False).reset_index(drop=True)
        df_example_sorted = df_example.sort_values(
            by='highlight', ascending=False).reset_index(drop=True)

        return df_our_sorted, df_example_sorted

    def create_results_section(self, df_sorted, title, column):
        """創建一個帶滾動條的結果框架，並填充匹配結果"""
        frame = tk.Frame(self.results_frame)
        frame.grid(row=0, column=column, padx=10, pady=10, sticky="n")

        tk.Label(frame, text=title, font=("Arial", 12, "bold")).pack()

        canvas = tk.Canvas(frame, width=600, height=300)
        scrollbar = tk.Scrollbar(
            frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.populate_result_rows(scrollable_frame, df_sorted)

    def populate_result_rows(self, frame, df_sorted):
        """填充每個匹配結果到分隔的行中"""
        for i, row in df_sorted.iterrows():
            if row['Product Title'] == '':
                continue

            bg_color = "yellow" if row['highlight'] else None

            row_frame = tk.Frame(frame, bd=1, relief="solid", padx=5, pady=5)
            row_frame.pack(fill="x", pady=2)

            # 顯示 Index
            index_label = tk.Label(
                row_frame,
                text=f"{i + 1}",
                width=5,
                anchor="w",
                background=bg_color)
            index_label.grid(row=0, column=0, sticky="w")

            # 顯示 MOMO Title
            momotitle_label = tk.Label(
                row_frame,
                text=row['MOMO Title'],
                wraplength=200,
                anchor="w",
                width=30)
            momotitle_label.grid(row=0, column=1, sticky="w")

            # 顯示 Product Title
            producttitle_label = tk.Label(
                row_frame,
                text=row['Product Title'],
                wraplength=200,
                anchor="w",
                width=30)
            producttitle_label.grid(row=0, column=2, sticky="w")

            # 顯示 Similarity
            similarity_label = tk.Label(
                row_frame,
                text=f"{row['Similarity']:.2f}",
                width=15,
                anchor="w")
            similarity_label.grid(row=0, column=3, sticky="w")

            # 高亮顯示匹配項
            # if row['highlight']:
            #     row_frame.config(bg="yellow")

    def plot_similarity_distribution(self):
        """繪製 Example Matching Results 和 Our Matching Results Similarity 分布圖，根據匹配情況標示不同顏色"""
        query = self.selected_query.get()
        file_v1 = os.path.join(self.folder_v1, f"{query}.csv")
        file_example = os.path.join(self.folder_example, f"{query}.csv")

        if os.path.exists(file_v1) and os.path.exists(file_example):
            # 讀取 CSV 檔案
            df_v1 = pd.read_csv(file_v1)
            df_example = pd.read_csv(file_example)

            # 排除 Similarity 為 NaN 的行
            df_v1 = df_v1.dropna(subset=['Similarity'])
            df_example = df_example.dropna(subset=['Similarity'])

            # 創建標示欄位
            df_v1['highlight'] = df_v1.apply(
                lambda row: row['MOMO Title'] in df_example['MOMO Title'].values and row['Product Title'] in df_example['Product Title'].values, axis=1)
            df_example['highlight'] = df_example.apply(
                lambda row: row['MOMO Title'] in df_v1['MOMO Title'].values and row['Product Title'] in df_v1['Product Title'].values, axis=1)

            # 匹配與不匹配的相似度
            matched_v1 = df_v1[df_v1['highlight'] == True]['Similarity']
            unmatched_v1 = df_v1[df_v1['highlight'] == False]['Similarity']
            matched_example = df_example[df_example['highlight']
                                         == True]['Similarity']
            unmatched_example = df_example[df_example['highlight']
                                           == False]['Similarity']

            print(f'len(unmatched_example): {len(matched_example)}')

            # 繪製 Similarity 分布圖
            plt.figure(figsize=(10, 6))
            plt.hist(matched_v1, bins=30, alpha=0.5,
                     label='Matched Our Results', color='red')
            plt.hist(unmatched_v1, bins=30, alpha=0.5,
                     label='Unmatched Our Results', color='blue')
            plt.hist(matched_example, bins=30, alpha=0.5,
                     label='Matched Example Results', color='orange')
            plt.hist(unmatched_example, bins=30, alpha=0.5,
                     label='Unmatched Example Results', color='green')
            plt.xlim(0, 1)
            plt.title('Similarity Distribution Comparison')
            plt.xlabel('Similarity')
            plt.ylabel('Frequency')
            plt.legend(loc='upper right')

            plt.show()
        else:
            messagebox.showerror(
                "Error", f"Files for query '{query}' do not exist.")


# 啟動 UI
root = tk.Tk()
app = MatchingResultsApp(
    root,
    queries_file='./queries/M11207424_queries.txt',  # 假設搜尋詞在這個文件中
    folder_v1='matching_results - v1.0',  # v1.0 資料夾
    folder_example='matching_results - Example'  # Example 資料夾
)
root.mainloop()
