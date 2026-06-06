import sys
import numpy as np
import pandas as pd
from enum import Enum
import matplotlib.pyplot as plt
from PySide6 import *
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

"""
等待读取文件
|
|---预览文件
    |
    |---浏览统计信息
    |---生成图像
    |---高级统计方法
        |---帮助
        |---统计结果

"""

plt.rcParams['font.family'] = 'SimHei'  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class AppState(Enum):
    NO_DATA = 11

    PREVIEW = 21

    PRIMARY_STATISTICS = 31
    GENERATE_PLOTS = 32
    ADVANCE_STATISTICS = 33

    HELP = 41
    RESULT = 42


class MyWindow(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.selected_x = []
        self.selected_y = []
        self.setWindowTitle("数据处理软件")
        self.resize(1000, 600)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.init_no_data_ui()

    def clear_widgets(self):
        while self.layout.count():
            item = self.layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)

    def init_no_data_ui(self):
        left_widget = QWidget()
        right_widget = QWidget()
        left_layout = QHBoxLayout(left_widget)
        right_layout = QHBoxLayout(right_widget)

        btn_open_file = QPushButton("读取excel/csv")
        btn_open_file.clicked.connect(self.btn_open_file_action)
        tips_label = QLabel("暂无数据")
        tips_label.setAlignment(Qt.AlignCenter)

        left_layout.addWidget(btn_open_file)
        right_layout.addWidget(tips_label)

        splitter = QSplitter()
        splitter.setStyleSheet("""
            QSplitter::handle {
                background: #ccc;
                height: 6px;
            }
        """)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)

        self.layout.addWidget(splitter)

    def btn_open_file_action(self):
        self.app.df = self.open_file()
        self.app.status = AppState.PREVIEW
        self.clear_widgets()
        self.init_preview_ui()

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择文件",
            "",
            "Excel Files (*.xlsx *.xls);;CSV Files (*.csv)"
        )

        if not file_path:
            return pd.DataFrame()

        if file_path.endswith(".csv"):
            return pd.read_csv(file_path)
        else:
            return pd.read_excel(file_path)

    def init_preview_ui(self):
        left_widget = QWidget()
        right_preview_widget = QWidget()
        right_model_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        right_preview_layout = QHBoxLayout(right_preview_widget)
        right_model_layout = QHBoxLayout(right_model_widget)

        btn_open_file = QPushButton("读取excel/csv")
        btn_open_file.clicked.connect(self.open_file)

        preview_table = QTableWidget()
        preview_table.setRowCount(len(self.app.df))
        preview_table.setColumnCount(len(self.app.df.columns))
        preview_table.setHorizontalHeaderLabels(self.app.df.columns.astype(str))
        for row in range(len(self.app.df)):
            for col in range(len(self.app.df.columns)):
                item = QTableWidgetItem(str(self.app.df.iloc[row, col]))
                preview_table.setItem(row, col, item)
        preview_table.resizeColumnsToContents()

        btn_statistics = QPushButton("浏览统计数据")
        btn_statistics.clicked.connect(self.btn_statistics_action)
        btn_plot = QPushButton("图像绘制")
        btn_plot.clicked.connect(self.btn_plot_action)
        btn_advance = QPushButton("高级统计方法")
        btn_advance.clicked.connect(self.btn_advance_action)

        left_layout.addWidget(btn_open_file)
        right_preview_layout.addWidget(preview_table)
        right_model_layout.addWidget(btn_statistics)
        right_model_layout.addWidget(btn_plot)
        right_model_layout.addWidget(btn_advance)

        splitter = QSplitter()
        splitter.setStyleSheet("""
                    QSplitter::handle {
                        background: #ccc;
                        height: 6px;
                    }
                """)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.addWidget(right_model_widget)
        right_layout.addWidget(right_preview_widget)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)

        self.layout.addWidget(splitter)

    def btn_statistics_action(self):
        self.app.status = AppState.PRIMARY_STATISTICS
        self.clear_widgets()
        self.init_primary_statistics_ui()

    def btn_plot_action(self):
        self.selected_x = []
        self.selected_y = []
        self.app.status = AppState.GENERATE_PLOTS
        self.clear_widgets()
        self.init_generate_plot_ui()

    def btn_advance_action(self):
        self.app.status = AppState.ADVANCE_STATISTICS
        self.clear_widgets()
        self.init_advance_statistics_ui()

    def init_primary_statistics_ui(self):
        left_widget = QWidget()
        right_preview_widget = QWidget()
        right_model_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        right_preview_layout = QHBoxLayout(right_preview_widget)
        right_model_layout = QHBoxLayout(right_model_widget)

        btn_open_file = QPushButton("读取excel/csv")
        btn_open_file.clicked.connect(self.open_file)
        btn_back_to_preview = QPushButton("返回预览")
        btn_back_to_preview.clicked.connect(self.btn_back_to_preview_action)

        btn_statistics = QPushButton("浏览统计数据")
        btn_statistics.clicked.connect(self.btn_statistics_action)
        btn_plot = QPushButton("图像绘制")
        btn_plot.clicked.connect(self.btn_plot_action)
        btn_advance = QPushButton("高级统计方法")
        btn_advance.clicked.connect(self.btn_advance_action)

        self.get_statistics(self.app.df)

        left_layout.addWidget(btn_open_file)
        left_layout.addWidget(btn_back_to_preview)
        left_layout.addStretch(1)

        right_model_layout.addWidget(btn_statistics)
        right_model_layout.addWidget(btn_plot)
        right_model_layout.addWidget(btn_advance)
        preview_table = QTableWidget()
        stats_df = self.get_statistics(self.app.df)
        preview_table.setRowCount(len(stats_df))
        preview_table.setColumnCount(len(stats_df.columns))
        preview_table.setHorizontalHeaderLabels(stats_df.columns.astype(str))
        for row in range(len(stats_df)):
            for col in range(len(stats_df.columns)):
                item = QTableWidgetItem(str(stats_df.iloc[row, col]))
                preview_table.setItem(row, col, item)
        preview_table.resizeColumnsToContents()
        right_preview_layout.addWidget(preview_table)

        splitter = QSplitter()
        splitter.setStyleSheet("""
                            QSplitter::handle {
                                background: #ccc;
                                height: 6px;
                            }
                        """)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.addWidget(right_model_widget)
        right_layout.addWidget(right_preview_widget)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)

        self.layout.addWidget(splitter)

    def btn_back_to_preview_action(self):
        self.app.status = AppState.PREVIEW
        self.clear_widgets()
        self.init_preview_ui()

    def get_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        stats = []
        n_rows = df.shape[0]
        n_cols = df.shape[1]
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                s = df[col]
                stats.append({
                    "列名": col,
                    "行数": n_rows,
                    "列数": n_cols,
                    "空值": s.isnull().sum(),
                    "平均数": s.mean(),
                    "最小值": s.min(),
                    "最大值": s.max(),
                    "方差": s.var(),
                    "标准差": s.std(),
                    "中位数": s.median(),
                    "前四分之一位数": s.quantile(0.25),
                    "后四分之一位数": s.quantile(0.75)
                })
        return pd.DataFrame(stats)

    def init_generate_plot_ui(self):
        left_widget = QWidget()
        right_preview_widget = QWidget()
        right_model_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        right_preview_layout = QVBoxLayout(right_preview_widget)
        right_model_layout = QHBoxLayout(right_model_widget)

        btn_open_file = QPushButton("读取excel/csv")
        btn_open_file.clicked.connect(self.btn_open_file_action)
        btn_back_to_preview = QPushButton("返回预览")
        btn_back_to_preview.clicked.connect(self.btn_back_to_preview_action)
        btn_generate_line = QPushButton("生成折线图")
        btn_generate_line.clicked.connect(self.btn_generate_line_action)
        btn_generate_scatter = QPushButton("生成散点图")
        btn_generate_scatter.clicked.connect(self.btn_generate_scatter_action)
        btn_generate_bar = QPushButton("生成柱状图")
        btn_generate_bar.clicked.connect(self.btn_generate_bar_action)

        btn_statistics = QPushButton("浏览统计数据")
        btn_statistics.clicked.connect(self.btn_statistics_action)
        btn_plot = QPushButton("图像绘制")
        btn_plot.clicked.connect(self.btn_plot_action)
        btn_advance = QPushButton("高级统计方法")
        btn_advance.clicked.connect(self.btn_advance_action)

        self.get_statistics(self.app.df)

        left_layout.addWidget(btn_open_file)
        left_layout.addWidget(btn_back_to_preview)
        left_layout.addStretch(1)
        left_layout.addWidget(btn_generate_line)
        left_layout.addWidget(btn_generate_scatter)
        left_layout.addWidget(btn_generate_bar)

        right_model_layout.addWidget(btn_statistics)
        right_model_layout.addWidget(btn_plot)
        right_model_layout.addWidget(btn_advance)
        numeric_df = self.app.df.select_dtypes(include=[np.number])
        x_grid_layout = QGridLayout()
        y_grid_layout = QGridLayout()
        cols_per_row = 4  # 一行4个按钮

        x_tip_label = QLabel("选择X列")
        y_tip_label = QLabel("选择Y列")

        for i, col in enumerate(numeric_df.columns):
            if pd.api.types.is_numeric_dtype(numeric_df[col]):
                btn_x = QPushButton(col)

                def action(_=None, c=col):
                    if c not in self.selected_x:
                        self.selected_x.append(c)
                        x_tip_label.setText(f"已选择的X列:{self.selected_x}")
                    elif c in self.selected_x:
                        self.selected_x.remove(c)
                        x_tip_label.setText(f"已选择的X列:{self.selected_x}")

                btn_x.clicked.connect(action)
                row = i // cols_per_row * 2 + 1
                col_pos = i % cols_per_row
                x_grid_layout.addWidget(btn_x, row, col_pos)
        for i, col in enumerate(numeric_df.columns):
            if pd.api.types.is_numeric_dtype(numeric_df[col]):
                btn_y = QPushButton(col)

                def action(_=None, c=col):
                    if c not in self.selected_y:
                        self.selected_y.append(c)
                        y_tip_label.setText(f"已选择的Y列:{self.selected_y}")
                    elif c in self.selected_y:
                        self.selected_y.remove(c)
                        y_tip_label.setText(f"已选择的Y列:{self.selected_y}")

                btn_y.clicked.connect(action)
                row = i // cols_per_row * 2 + 1
                col_pos = i % cols_per_row
                y_grid_layout.addWidget(btn_y, row + 1, col_pos)

        right_preview_layout.addWidget(x_tip_label)
        right_preview_layout.addLayout(x_grid_layout)
        right_preview_layout.addWidget(y_tip_label)
        right_preview_layout.addLayout(y_grid_layout)

        splitter = QSplitter()
        splitter.setStyleSheet("""
                            QSplitter::handle {
                                background: #ccc;
                                height: 6px;
                            }
                        """)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.addWidget(right_model_widget)
        right_layout.addWidget(right_preview_widget)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)

        self.layout.addWidget(splitter)

    def generate_plot(self, kind="line"):
        try:
            if self.selected_x is None or self.selected_y is None:
                pass
            else:
                m = len(self.selected_y)
                n = len(self.selected_x)
                fig, axes = plt.subplots(
                    m, n,
                    figsize=(10, 6),
                    sharex=False, sharey=False
                )

                # 如果只有一行或一列，axes 不是二维数组，统一处理
                if m == 1 and n == 1:
                    axes = [[axes]]
                elif m == 1:
                    axes = [axes]
                elif n == 1:
                    axes = [[ax] for ax in axes]

                for i, y_col in enumerate(self.selected_y):
                    for j, x_col in enumerate(self.selected_x):
                        ax = axes[i][j]
                        if kind == "line":
                            ax.plot(self.app.df[x_col], self.app.df[y_col], marker="o")
                        elif kind == "scatter":
                            ax.scatter(self.app.df[x_col], self.app.df[y_col], s=20)
                        elif kind == "bar":
                            ax.bar(self.app.df[x_col], self.app.df[y_col])
                        ax.set_xlabel(x_col)
                        ax.set_ylabel(y_col)
                        ax.set_title(f"{y_col} vs {x_col}")

                plt.tight_layout()
                plt.show()
        except ValueError as e:
            print(e)
            print("请选取需要的列再生成图像")

    def btn_generate_line_action(self):
        self.generate_plot(kind="line")

    def btn_generate_scatter_action(self):
        self.generate_plot(kind="scatter")

    def btn_generate_bar_action(self):
        self.generate_plot(kind="bar")

    def init_advance_statistics_ui(self):
        self.selected_x = []
        self.selected_y = []
        left_widget = QWidget()
        right_preview_widget = QWidget()
        right_model_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        right_preview_layout = QVBoxLayout(right_preview_widget)
        right_model_layout = QHBoxLayout(right_model_widget)

        btn_open_file = QPushButton("读取excel/csv")
        btn_open_file.clicked.connect(self.btn_open_file_action)
        btn_back_to_preview = QPushButton("返回预览")
        btn_back_to_preview.clicked.connect(self.btn_back_to_preview_action)
        btn_help = QPushButton("帮助")
        btn_help.clicked.connect(self.btn_help_action)
        btn_linear_regression = QPushButton("线性回归方法")
        btn_linear_regression.clicked.connect(self.btn_linear_regression_action)
        btn_random_forest = QPushButton("随机森林方法")
        btn_random_forest.clicked.connect(self.btn_random_forest_action)

        btn_statistics = QPushButton("浏览统计数据")
        btn_statistics.clicked.connect(self.btn_statistics_action)
        btn_plot = QPushButton("图像绘制")
        btn_plot.clicked.connect(self.btn_plot_action)
        btn_advance = QPushButton("高级统计方法")
        btn_advance.clicked.connect(self.btn_advance_action)

        self.get_statistics(self.app.df)

        left_layout.addWidget(btn_open_file)
        left_layout.addWidget(btn_back_to_preview)
        left_layout.addStretch(1)
        left_layout.addWidget(btn_help)
        left_layout.addWidget(btn_linear_regression)
        left_layout.addWidget(btn_random_forest)

        right_model_layout.addWidget(btn_statistics)
        right_model_layout.addWidget(btn_plot)
        right_model_layout.addWidget(btn_advance)
        numeric_df = self.app.df.select_dtypes(include=[np.number])
        x_grid_layout = QGridLayout()
        y_grid_layout = QGridLayout()
        cols_per_row = 4  # 一行4个按钮

        x_tip_label = QLabel("选择X列")
        y_tip_label = QLabel("选择Y列")

        for i, col in enumerate(numeric_df.columns):
            if pd.api.types.is_numeric_dtype(numeric_df[col]):
                btn_x = QPushButton(col)

                def action(_=None, c=col):
                    if c not in self.selected_x:
                        self.selected_x.append(c)
                        x_tip_label.setText(f"已选择的X列:{self.selected_x}")
                    elif c in self.selected_x:
                        self.selected_x.remove(c)
                        x_tip_label.setText(f"已选择的X列:{self.selected_x}")

                btn_x.clicked.connect(action)
                row = i // cols_per_row * 2 + 1
                col_pos = i % cols_per_row
                x_grid_layout.addWidget(btn_x, row, col_pos)
        for i, col in enumerate(numeric_df.columns):
            if pd.api.types.is_numeric_dtype(numeric_df[col]):
                btn_y = QPushButton(col)

                def action(_=None, c=col):
                    if c not in self.selected_y:
                        self.selected_y.append(c)
                        y_tip_label.setText(f"已选择的Y列:{self.selected_y}")
                    elif c in self.selected_y:
                        self.selected_y.remove(c)
                        y_tip_label.setText(f"已选择的Y列:{self.selected_y}")

                btn_y.clicked.connect(action)
                row = i // cols_per_row * 2 + 1
                col_pos = i % cols_per_row
                y_grid_layout.addWidget(btn_y, row + 1, col_pos)

        right_preview_layout.addWidget(x_tip_label)
        right_preview_layout.addLayout(x_grid_layout)
        right_preview_layout.addWidget(y_tip_label)
        right_preview_layout.addLayout(y_grid_layout)

        splitter = QSplitter()
        splitter.setStyleSheet("""
                                    QSplitter::handle {
                                        background: #ccc;
                                        height: 6px;
                                    }
                                """)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.addWidget(right_model_widget)
        right_layout.addWidget(right_preview_widget)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)

        self.layout.addWidget(splitter)

    def advance_statistics(self, method="linear_regression"):
        if method == "linear_regression":
            x = self.app.df[self.selected_x]
            y = self.app.df[self.selected_y[0]]
            X = sm.add_constant(x)
            model = sm.OLS(y, X)
            results = model.fit()
            return str(results.summary())
        elif method == "random_forest":
            X = self.app.df[self.selected_x].dropna()
            y = self.app.df[self.selected_y[0]].loc[X.index]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )

            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                random_state=42
            )
            model.fit(X_train, y_train)

            # 预测
            y_pred = model.predict(X_test)
            y_train_pred = model.predict(X_train)

            # 评价指标
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = mse ** 0.5
            mae = mean_absolute_error(y_test, y_pred)

            train_r2 = r2_score(y_train, y_train_pred)
            train_mse = mean_squared_error(y_train, y_train_pred)
            train_rmse = train_mse ** 0.5

            # 结果文本
            result = f"""
            ╔══════════════════════════════════════════════════════════════╗
            ║                  随机森林回归模型评估报告                   ║
            ╚══════════════════════════════════════════════════════════════╝

            数据概况
            ------------------------------------------------------------
            总样本数         : {len(X)}
            训练集样本数     : {len(X_train)}
            测试集样本数     : {len(X_test)}
            特征数           : {X.shape[1]}
            测试集比例       : 30%
            随机种子         : 42

            模型参数
            ------------------------------------------------------------
            模型类型         : RandomForestRegressor
            树的数量         : 100
            最大深度         : {'None (不限制)' if model.max_depth is None else model.max_depth}
            最小叶子节点     : {model.min_samples_leaf}
            最小分裂样本数   : {model.min_samples_split}
            随机特征数       : {'auto'}

            测试集性能
            ------------------------------------------------------------
            R² 决定系数      : {r2:.4f}
            MSE 均方误差     : {mse:.4f}
            RMSE 均方根误差  : {rmse:.4f}
            MAE 平均绝对误差 : {mae:.4f}

            训练集性能
            ------------------------------------------------------------
            R²               : {train_r2:.4f}
            RMSE             : {train_rmse:.4f}

            过拟合判断
            ------------------------------------------------------------
            训练 / 测试 R²差 : {abs(train_r2 - r2):.4f}
            {"可能存在过拟合" if train_r2 - r2 > 0.1 else "未见明显过拟合"}

            特征重要性
            ------------------------------------------------------------
            """

            for col, imp in sorted(
                    zip(self.selected_x, model.feature_importances_),
                    key=lambda x: x[1],
                    reverse=True
            ):
                result += f"{col:25s} : {imp:.6f}\n"

            return result
        return None

    def btn_help_action(self):
        self.app.status = AppState.HELP
        self.clear_widgets()
        self.init_help_ui()

    def btn_linear_regression_action(self):
        if len(self.selected_x) > 0 and len(self.selected_y) > 0:
            self.app.status = AppState.RESULT
            self.clear_widgets()
            self.init_result_ui(method="linear_regression")
        else:
            print("警告：请选取所需要的列后再进行统计操作")

    def btn_random_forest_action(self):
        if len(self.selected_x) > 0 and len(self.selected_y) > 0:
            self.app.status = AppState.RESULT
            self.clear_widgets()
            self.init_result_ui(method="random_forest")
        else:
            print("警告：请选取所需要的列后再进行统计操作")

    def init_help_ui(self):
        left_widget = QWidget()
        right_preview_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        right_preview_layout = QVBoxLayout(right_preview_widget)

        btn_open_file = QPushButton("读取excel/csv")
        btn_open_file.clicked.connect(self.btn_open_file_action)
        btn_back_to_preview = QPushButton("返回高级统计方法")
        btn_back_to_preview.clicked.connect(self.btn_advance_action)

        help_text = """
        ==========   帮助   ==========
        线性拟合部分参数意义：
        Dep. Variable： 被解释变量
        Method： Least Squares（最小化残差平方和来估计参数）
        No. Observations： 样本量
        Df Model： 模型中自变量的个数
        R-squared： 模型可以解释多少比例的现象（0~1）
        Adj. R-squared： 调整后的R，减少了自变量数量的影响
        F-statistic & Prob(F)： 检验 模型整体是否显著（p越小越显著）
        Log-Likelihood： 对数似然值，用于比较模型好坏（越小越好）
        const： 截距
        自变量名： 回归系数（斜率）
        std err： 标准差
        Omnibus / Prob(Omnibus)： 检验残差是否正态分布（p = 0 → 残差非正态）
        Skewness： 偏度
        Kurtosis： 峰度
        Cond. No.：多重共线性指标（小于100时一般安全）    
        """
        # 滚动区域
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        text_label = QLabel()
        text_label.setWordWrap(True)  # 自动换行
        text_label.setTextFormat(Qt.PlainText)
        text_label.setFont(QFont("Microsoft YaHei", 10))
        text_label.setText(help_text)

        scroll.setWidget(text_label)

        right_preview_layout.addWidget(scroll)

        left_layout.addWidget(btn_open_file)
        left_layout.addWidget(btn_back_to_preview)
        left_layout.addStretch(1)

        splitter = QSplitter()
        splitter.setStyleSheet("""
                                                    QSplitter::handle {
                                                        background: #ccc;
                                                        height: 6px;
                                                    }
                                                """)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.addWidget(right_preview_widget)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)

        self.layout.addWidget(splitter)

    def init_result_ui(self, method="linear_regression"):
        left_widget = QWidget()
        right_preview_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        right_preview_layout = QVBoxLayout(right_preview_widget)

        btn_open_file = QPushButton("读取excel/csv")
        btn_open_file.clicked.connect(self.btn_open_file_action)
        btn_back_to_preview = QPushButton("返回高级统计方法")
        btn_back_to_preview.clicked.connect(self.btn_advance_action)
        btn_help = QPushButton("帮助")
        btn_help.clicked.connect(self.btn_help_action)

        result = self.advance_statistics(method=method)
        result_text = QTextEdit()
        result_text.setReadOnly(True)
        result_text.setPlainText(result)
        right_preview_layout.addWidget(result_text)

        left_layout.addWidget(btn_open_file)
        left_layout.addWidget(btn_back_to_preview)
        left_layout.addStretch(1)
        left_layout.addWidget(btn_help)

        splitter = QSplitter()
        splitter.setStyleSheet("""
                                            QSplitter::handle {
                                                background: #ccc;
                                                height: 6px;
                                            }
                                        """)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.addWidget(right_preview_widget)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)

        self.layout.addWidget(splitter)


class App:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.window = MyWindow(self)
        self.state = AppState.NO_DATA
        self.df = pd.DataFrame()


if __name__ == "__main__":
    app = App()
    app.window.show()
    sys.exit(app.app.exec())

