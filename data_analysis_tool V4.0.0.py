import sys
import json
import os
import numpy as np
import pandas as pd
import matplotlib
# 关键：告诉matplotlib使用Qt5Agg后端，使其能与PySide6窗口交互
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
# 导入PySide6的子模块
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

BTN_FOR_CHOOSE_IN_LINE = 4  #控制每行可用于选择的按钮
CONFIG_FILE_PATH = "data_analysis_tool_config.json"

plt.rcParams['font.family'] = 'SimHei'  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def load_config():
    """从JSON文件加载配置"""
    default_config = {
        "last_file_path": "",
        "window_width": 1200,
        "window_height": 800,
        "window_x": None,
        "window_y": None
    }

    if not os.path.exists(CONFIG_FILE_PATH):
        return default_config

    try:
        with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 确保所有必需的键都存在
        for key in default_config:
            if key not in config:
                config[key] = default_config[key]

        return config
    except (json.JSONDecodeError, IOError):
        return default_config

def save_config(config):
    """将配置保存到JSON文件"""
    try:
        with open(CONFIG_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    except IOError as e:
        print(f"保存配置文件失败: {e}")

def get_toolbar(page):
    # 顶部工具栏
    toolbar = QHBoxLayout()
    btn_open = QPushButton("读取excel/csv")
    btn_open.clicked.connect(page._on_open_file)

    btn_stats = QPushButton("浏览统计数据")
    btn_stats.clicked.connect(page.go_to_statistics.emit)

    btn_plot = QPushButton("图像绘制")
    btn_plot.clicked.connect(page.go_to_plot.emit)

    btn_advance = QPushButton("高级统计方法")
    btn_advance.clicked.connect(page.go_to_advance.emit)

    btn_preview = QPushButton("预览界面")
    btn_preview.clicked.connect(page.go_to_preview.emit)

    toolbar.addWidget(btn_open)
    toolbar.addWidget(btn_stats)
    toolbar.addWidget(btn_plot)
    toolbar.addWidget(btn_advance)
    toolbar.addWidget(btn_preview)
    toolbar.addStretch()

    return toolbar

def open_file(parent_widget, analyzer):
    """独立的文件打开函数

    Args:
        parent_widget: 父窗口组件，用于显示对话框
        analyzer: DataAnalyzer实例

    Returns:
        bool: 是否成功打开文件
    """
    # 读取配置获取上次的文件路径
    config = load_config()
    last_path = config.get("last_file_path", "")

    # 如果上次路径存在，使用其目录作为初始目录
    initial_dir = os.path.dirname(last_path) if last_path and os.path.exists(os.path.dirname(last_path)) else ""

    file_path, _ = QFileDialog.getOpenFileName(
        parent_widget, "选择文件", initial_dir,
        "Excel Files (*.xlsx *.xls);;CSV Files (*.csv)"
    )

    if file_path and analyzer.load_file(file_path):
        # 保存文件路径到配置
        config["last_file_path"] = file_path
        save_config(config)
        return True

    return False



class DataAnalyzer:
    """DataAnalyzer类,负责所有数据分析和建模的业务逻辑"""
    def __init__(self):
        self.df = pd.DataFrame()
        self.selected_x = []
        self.selected_y = []
        self.color_selected = None
        self.cleaning_history = []

    def load_file(self, file_path: str) -> bool:
        """加载文件，成功返回True，失败返回False"""
        try:
            if file_path.endswith('.csv'):
                self.df = pd.read_csv(file_path)
            else:
                engine = 'xlrd' if file_path.endswith('.xls') else 'openpyxl'
                self.df = pd.read_excel(file_path, engine=engine)
            return True
        except Exception as e:
            print(f"文件读取失败: {e}")
            return False

    def has_data(self) -> bool:
        """检查表格是否为空"""
        return not self.df.empty

    def get_numeric_columns(self) -> list:
        """获取纯数值列"""
        return self.df.select_dtypes(include=[np.number]).columns.tolist()

    def get_statistics(self) -> pd.DataFrame:
        """计算数值列的统计信息"""
        stats = []
        n_rows = self.df.shape[0]
        n_cols = self.df.shape[1]

        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                s = self.df[col]
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

    def linear_regression(self) -> str:
        """执行线性回归，返回结果摘要"""
        if not self.selected_x or not self.selected_y:
            return "请先选择X和Y列"

        x = self.df[self.selected_x]
        y = self.df[self.selected_y[0]]
        X = sm.add_constant(x)  #添加全1常数列，防止过原点
        model = sm.OLS(y, X)
        results = model.fit()
        return str(results.summary())

    def random_forest(self) -> str:
        """执行随机森林回归，返回格式化结果"""
        if not self.selected_x or not self.selected_y:
            return "请先选择X和Y列"

        X = self.df[self.selected_x].dropna()
        y = self.df[self.selected_y[0]].loc[X.index]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        model = RandomForestRegressor(
            n_estimators=100, max_depth=None, random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)

        # 计算指标
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_test, y_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        train_rmse = mean_squared_error(y_train, y_train_pred) ** 0.5

        # 格式化输出
        result = [
            "=" * 60,
            "随机森林回归模型评估报告",
            "=" * 60,
            "",
            f"总样本数: {len(X)}",
            f"训练集样本数: {len(X_train)}",
            f"测试集样本数: {len(X_test)}",
            f"特征数: {X.shape[1]}",
            "",
            "--- 测试集性能 ---",
            f"R² 决定系数: {r2:.4f}",
            f"MSE 均方误差: {mse:.4f}",
            f"RMSE 均方根误差: {rmse:.4f}",
            f"MAE 平均绝对误差: {mae:.4f}",
            "",
            "--- 训练集性能 ---",
            f"R²: {train_r2:.4f}",
            f"RMSE: {train_rmse:.4f}",
            "",
            "--- 特征重要性 ---",
        ]

        for col, imp in sorted(
                zip(self.selected_x, model.feature_importances_),
                key=lambda x: x[1], reverse=True
        ):
            result.append(f"  {col}: {imp:.6f}")

        return "\n".join(result)

    def clean_missing_values(self, strategy='drop', fill_value=None):
        """处理缺失值

        Args:
            strategy: 'drop' - 删除含缺失值的行
                      'fill_mean' - 用均值填充
                      'fill_median' - 用中位数填充
                      'fill_mode' - 用众数填充
                      'fill_value' - 用指定值填充
            fill_value: 当strategy='fill_value'时的填充值
        """
        if self.df.empty:
            return False, "没有数据可清理"

        original_shape = self.df.shape
        missing_before = self.df.isnull().sum().sum()

        try:
            if strategy == 'drop':
                self.df = self.df.dropna()
                record_msg = f"删除含缺失值的行：{original_shape[0]}行 → {self.df.shape[0]}行"

            elif strategy == 'fill_mean':
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
                record_msg = f"用均值填充缺失值（共{missing_before}个）"

            elif strategy == 'fill_median':
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
                record_msg = f"用中位数填充缺失值（共{missing_before}个）"

            elif strategy == 'fill_mode':
                for col in self.df.columns:
                    mode_val = self.df[col].mode()
                    if not mode_val.empty:
                        self.df[col] = self.df[col].fillna(mode_val[0])
                record_msg = f"用众数填充缺失值（共{missing_before}个）"

            elif strategy == 'fill_value':
                if fill_value is None:
                    return False, "请指定填充值"
                self.df = self.df.fillna(fill_value)
                record_msg = f"用指定值({fill_value})填充缺失值（共{missing_before}个）"

            else:
                return False, f"未知策略: {strategy}"

            missing_after = self.df.isnull().sum().sum()
            self.cleaning_history.append(record_msg)

            return True, f"处理完成！{record_msg}\n剩余缺失值：{missing_after}个"

        except Exception as e:
            return False, f"处理缺失值时出错：{str(e)}"

    def remove_duplicates(self, subset=None, keep='first'):
        """删除重复行

        Args:
            subset: 列名列表，用于检测重复的列（None表示所有列）
            keep: 'first' - 保留第一条
                  'last' - 保留最后一条
                  False - 删除所有重复行
        """
        if self.df.empty:
            return False, "没有数据可清理"

        original_shape = self.df.shape

        try:
            duplicates_count = self.df.duplicated(subset=subset, keep=False).sum()

            if duplicates_count == 0:
                return False, "未发现重复行"

            self.df = self.df.drop_duplicates(subset=subset, keep=keep)

            removed = original_shape[0] - self.df.shape[0]
            record_msg = f"删除重复行：移除{removed}行（{original_shape[0]}行 → {self.df.shape[0]}行）"
            self.cleaning_history.append(record_msg)

            return True, f"处理完成！{record_msg}"

        except Exception as e:
            return False, f"删除重复行时出错：{str(e)}"

    def remove_outliers(self, columns=None, method='iqr', threshold=1.5):
        """删除异常值

        Args:
            columns: 要处理的列名列表（None表示所有数值列）
            method: 'iqr' - 使用IQR方法
                    'zscore' - 使用Z-score方法
            threshold: IQR倍数或Z-score阈值
        """
        if self.df.empty:
            return False, "没有数据可清理"

        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()

        original_shape = self.df.shape
        total_removed = 0

        try:
            mask = pd.Series([True] * len(self.df))

            for col in columns:
                if col not in self.df.columns:
                    continue

                if not pd.api.types.is_numeric_dtype(self.df[col]):
                    continue

                if method == 'iqr':
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    col_mask = (self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)

                elif method == 'zscore':
                    mean = self.df[col].mean()
                    std = self.df[col].std()
                    if std == 0:
                        continue
                    z_scores = (self.df[col] - mean) / std
                    col_mask = z_scores.abs() <= threshold

                mask = mask & col_mask

            removed_count = (~mask).sum()
            if removed_count > 0:
                self.df = self.df[mask]
                total_removed = removed_count

            if total_removed > 0:
                record_msg = f"删除异常值：移除{total_removed}行（{original_shape[0]}行 → {self.df.shape[0]}行）"
                self.cleaning_history.append(record_msg)
                return True, f"处理完成！{record_msg}"
            else:
                return False, "未发现异常值"

        except Exception as e:
            return False, f"删除异常值时出错：{str(e)}"

    def reset_index(self):
        """重置索引"""
        self.df = self.df.reset_index(drop=True)
        self.cleaning_history.append("重置索引")
        return True, "索引已重置"

    def get_cleaning_summary(self):
        """获取清洗历史摘要"""
        if not self.cleaning_history:
            return "暂无清洗记录"

        summary = ["数据清洗历史："]
        summary.append("=" * 50)
        for i, record in enumerate(self.cleaning_history, 1):
            summary.append(f"{i}. {record}")
        summary.append("=" * 50)
        summary.append(f"当前数据：{self.df.shape[0]}行 × {self.df.shape[1]}列")

        return "\n".join(summary)

    def get_missing_info(self):
        """获取缺失值信息"""
        if self.df.empty:
            return "没有数据"

        missing_total = self.df.isnull().sum().sum()
        missing_by_col = self.df.isnull().sum()
        missing_pct = (self.df.isnull().sum() / len(self.df) * 100).round(2)

        info = [f"缺失值总数：{missing_total}", ""]
        info.append("各列缺失情况：")
        info.append("-" * 80)
        info.append(f"{'列名':<20} {'缺失数':<10} {'缺失率(%)':<10}")
        info.append("-" * 80)

        for col in self.df.columns:
            count = missing_by_col[col]
            pct = missing_pct[col]
            if count > 0:
                info.append(f"{col:<20} {count:<10} {pct:<10}")

        return "\n".join(info)

    def export_figure(self, fig, path):
        """导出 matplotlib 图像到文件"""
        try:
            ext = os.path.splitext(path)[1].lower()
            supported = ['.png', '.pdf', '.jpg', '.jpeg', '.svg']
            if ext not in supported:
                path += '.png'
            fig.savefig(path, dpi=200, bbox_inches='tight')
            return True, f"图像已导出至：{path}"
        except Exception as e:
            return False, f"导出图像失败：{str(e)}"

    def export_dataframe(self, df, path):
        """导出 DataFrame 到文件（Excel 或 CSV）"""
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext == '.xlsx':
                df.to_excel(path, index=False, engine='openpyxl')
            else:
                if ext != '.csv':
                    path += '.csv'
                df.to_csv(path, index=False, encoding='utf-8-sig')
            return True, f"数据已导出至：{path}"
        except Exception as e:
            return False, f"导出数据失败：{str(e)}"

    def export_text(self, text, path):
        """导出文本到文件"""
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext != '.txt':
                path += '.txt'
            with open(path, 'w', encoding='utf-8') as f:
                f.write(text)
            return True, f"文本已导出至：{path}"
        except Exception as e:
            return False, f"导出文本失败：{str(e)}"


class DataCleaningDialog(QDialog):
    """数据清洗对话框"""

    def __init__(self, analyzer, parent=None):
        super().__init__(parent)
        self.analyzer = analyzer
        self.setWindowTitle("数据清洗工具")
        self.setMinimumSize(600, 400)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # 标签页
        tab_widget = QTabWidget()

        # ===== 缺失值处理标签页 =====
        missing_tab = QWidget()
        missing_layout = QVBoxLayout(missing_tab)

        # 缺失值信息
        self.missing_info = QTextEdit()
        self.missing_info.setReadOnly(True)
        self.missing_info.setMaximumHeight(150)
        self.missing_info.setText(self.analyzer.get_missing_info())

        # 处理策略选择
        strategy_group = QGroupBox("处理策略")
        strategy_layout = QVBoxLayout(strategy_group)

        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems([
            "删除含缺失值的行",
            "用均值填充（数值列）",
            "用中位数填充（数值列）",
            "用众数填充",
            "用指定值填充"
        ])

        self.fill_value_input = QLineEdit()
        self.fill_value_input.setPlaceholderText("输入填充值（仅在选择'用指定值填充'时需要）")
        self.fill_value_input.setEnabled(False)

        self.strategy_combo.currentIndexChanged.connect(
            lambda idx: self.fill_value_input.setEnabled(idx == 4)
        )

        strategy_layout.addWidget(QLabel("选择处理策略："))
        strategy_layout.addWidget(self.strategy_combo)
        strategy_layout.addWidget(self.fill_value_input)

        btn_apply_missing = QPushButton("应用缺失值处理")
        btn_apply_missing.clicked.connect(self._apply_missing_handling)

        missing_layout.addWidget(self.missing_info)
        missing_layout.addWidget(strategy_group)
        missing_layout.addWidget(btn_apply_missing)
        missing_layout.addStretch()

        # ===== 重复值处理标签页 =====
        duplicate_tab = QWidget()
        duplicate_layout = QVBoxLayout(duplicate_tab)

        # 列选择
        dup_select_group = QGroupBox("选择检测重复的列")
        dup_select_layout = QVBoxLayout(dup_select_group)

        self.dup_column_list = QListWidget()
        self.dup_column_list.setSelectionMode(QListWidget.MultiSelection)
        for col in self.analyzer.df.columns:
            self.dup_column_list.addItem(col)

        dup_select_layout.addWidget(QLabel("按住Ctrl可选择多列（不选则检测所有列）："))
        dup_select_layout.addWidget(self.dup_column_list)

        # 保留策略
        keep_group = QGroupBox("保留策略")
        keep_layout = QVBoxLayout(keep_group)

        self.keep_combo = QComboBox()
        self.keep_combo.addItems(["保留第一条", "保留最后一条", "删除所有重复"])

        keep_layout.addWidget(self.keep_combo)

        btn_remove_duplicates = QPushButton("删除重复行")
        btn_remove_duplicates.clicked.connect(self._apply_duplicate_removal)

        duplicate_layout.addWidget(dup_select_group)
        duplicate_layout.addWidget(keep_group)
        duplicate_layout.addWidget(btn_remove_duplicates)
        duplicate_layout.addStretch()

        # ===== 异常值处理标签页 =====
        outlier_tab = QWidget()
        outlier_layout = QVBoxLayout(outlier_tab)

        # 列选择
        out_select_group = QGroupBox("选择要处理的数值列")
        out_select_layout = QVBoxLayout(out_select_group)

        self.out_column_list = QListWidget()
        self.out_column_list.setSelectionMode(QListWidget.MultiSelection)
        for col in self.analyzer.get_numeric_columns():
            self.out_column_list.addItem(col)

        out_select_layout.addWidget(QLabel("按住Ctrl可选择多列（不选则处理所有数值列）："))
        out_select_layout.addWidget(self.out_column_list)

        # 方法选择
        method_group = QGroupBox("异常值检测方法")
        method_layout = QVBoxLayout(method_group)

        self.method_combo = QComboBox()
        self.method_combo.addItems(["IQR方法（四分位距）", "Z-score方法"])

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.5, 10.0)
        self.threshold_spin.setValue(1.5)
        self.threshold_spin.setSingleStep(0.5)

        method_layout.addWidget(self.method_combo)
        method_layout.addWidget(QLabel("阈值："))
        method_layout.addWidget(self.threshold_spin)

        btn_remove_outliers = QPushButton("删除异常值")
        btn_remove_outliers.clicked.connect(self._apply_outlier_removal)

        outlier_layout.addWidget(out_select_group)
        outlier_layout.addWidget(method_group)
        outlier_layout.addWidget(btn_remove_outliers)
        outlier_layout.addStretch()

        # ===== 清洗历史标签页 =====
        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)

        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)
        self.history_text.setText(self.analyzer.get_cleaning_summary())

        btn_refresh_history = QPushButton("刷新历史")
        btn_refresh_history.clicked.connect(
            lambda: self.history_text.setText(self.analyzer.get_cleaning_summary())
        )

        history_layout.addWidget(self.history_text)
        history_layout.addWidget(btn_refresh_history)

        # 添加标签页
        tab_widget.addTab(missing_tab, "缺失值处理")
        tab_widget.addTab(duplicate_tab, "重复值处理")
        tab_widget.addTab(outlier_tab, "异常值处理")
        tab_widget.addTab(history_tab, "清洗历史")

        layout.addWidget(tab_widget)

        # 底部按钮
        button_box = QDialogButtonBox()
        btn_close = button_box.addButton("关闭", QDialogButtonBox.RejectRole)
        btn_close.clicked.connect(self.accept)

        layout.addWidget(button_box)

    def _apply_missing_handling(self):
        """应用缺失值处理"""
        strategy_map = {
            0: 'drop',
            1: 'fill_mean',
            2: 'fill_median',
            3: 'fill_mode',
            4: 'fill_value'
        }

        strategy = strategy_map[self.strategy_combo.currentIndex()]
        fill_value = None

        if strategy == 'fill_value':
            fill_text = self.fill_value_input.text().strip()
            if not fill_text:
                QMessageBox.warning(self, "警告", "请输入填充值")
                return
            # 尝试转换为数值
            try:
                fill_value = float(fill_text)
            except ValueError:
                fill_value = fill_text

        success, msg = self.analyzer.clean_missing_values(strategy, fill_value)

        if success:
            QMessageBox.information(self, "成功", msg)
            self.missing_info.setText(self.analyzer.get_missing_info())
            self.history_text.setText(self.analyzer.get_cleaning_summary())
        else:
            QMessageBox.warning(self, "提示", msg)

    def _apply_duplicate_removal(self):
        """应用重复值删除"""
        selected_items = self.dup_column_list.selectedItems()
        subset = [item.text() for item in selected_items] if selected_items else None

        keep_map = {0: 'first', 1: 'last', 2: False}
        keep = keep_map[self.keep_combo.currentIndex()]

        success, msg = self.analyzer.remove_duplicates(subset, keep)

        if success:
            QMessageBox.information(self, "成功", msg)
            self.history_text.setText(self.analyzer.get_cleaning_summary())
        else:
            QMessageBox.warning(self, "提示", msg)

    def _apply_outlier_removal(self):
        """应用异常值删除"""
        selected_items = self.out_column_list.selectedItems()
        columns = [item.text() for item in selected_items] if selected_items else None

        method_map = {0: 'iqr', 1: 'zscore'}
        method = method_map[self.method_combo.currentIndex()]
        threshold = self.threshold_spin.value()

        success, msg = self.analyzer.remove_outliers(columns, method, threshold)

        if success:
            QMessageBox.information(self, "成功", msg)
            self.history_text.setText(self.analyzer.get_cleaning_summary())
        else:
            QMessageBox.warning(self, "提示", msg)



class PreviewPage(QWidget):
    """数据预览页面"""

    # 定义信号，用于通知主窗口页面切换
    go_to_preview = Signal()
    go_to_statistics = Signal()
    go_to_plot = Signal()
    go_to_advance = Signal()
    file_opened = Signal()

    def __init__(self, analyzer):
        super().__init__()
        self.analyzer = analyzer
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # 顶部工具栏
        toolbar = get_toolbar(self)

        btn_clean = QPushButton("数据清洗")
        btn_clean.clicked.connect(self._open_cleaning_dialog)

        toolbar.addWidget(btn_clean)

        # 数据表格
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)  # 交替行颜色，提高可读性
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)  # 禁止编辑

        layout.addLayout(toolbar)
        layout.addWidget(self.table)

    def refresh(self):
        """刷新页面数据"""
        if not self.analyzer.has_data():
            return

        df = self.analyzer.df
        #处理表格
        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels(df.columns.astype(str))

        for row in range(len(df)):
            for col in range(len(df.columns)):
                item = QTableWidgetItem(str(df.iloc[row, col])) #填入数据
                self.table.setItem(row, col, item)

        self.table.resizeColumnsToContents()    #自动列宽

    def _on_open_file(self):
        if open_file(self, self.analyzer):
            self.refresh()
            self.file_opened.emit()

    def _open_cleaning_dialog(self):
        """打开数据清洗对话框"""
        if not self.analyzer.has_data():
            QMessageBox.warning(self, "警告", "请先加载数据文件！")
            return

        dialog = DataCleaningDialog(self.analyzer, self)
        if dialog.exec():
            # 刷新预览
            self.refresh()

    def get_color(col):
        """根据列值生成颜色映射"""
        max_val, min_val = col.max(skipna=True), col.min(skipna=True)
        norm = plt.Normalize(min_val, max_val)
        camp = plt.cm.plasma
        colors = camp(norm(col))
        return colors

class StatisticsPage(QWidget):
    """基础统计数据页面"""

    # 定义信号，用于通知主窗口页面切换
    go_to_preview = Signal()
    go_to_statistics = Signal()
    go_to_plot = Signal()
    go_to_advance = Signal()
    file_opened = Signal()

    def __init__(self, analyzer):
        super().__init__()
        self.analyzer = analyzer
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # 顶部工具栏
        toolbar = get_toolbar(self)

        # 导出按钮
        btn_export_stats = QPushButton("导出统计表")
        btn_export_stats.clicked.connect(self._export_statistics)
        toolbar.addStretch()
        toolbar.addWidget(btn_export_stats)

        # 统计信息表格
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)  # 交替行颜色，提高可读性
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)  # 禁止编辑


        layout.addLayout(toolbar)
        layout.addWidget(self.table)

    def refresh(self):
        """
        刷新统计数据表格。
        当页面被切换到前台时，由主窗口自动调用此方法。
        """
        if not self.analyzer.has_data():
            # 如果没有数据，清空表格并显示提示
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return

        # 获取统计数据
        stats_df = self.analyzer.get_statistics()

        # 如果统计数据为空（全是非数值列），也显示提示
        if stats_df.empty:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return

        # 设置表格行列
        rows = len(stats_df)
        cols = len(stats_df.columns)
        self.table.setRowCount(rows)
        self.table.setColumnCount(cols)

        # 设置表头
        headers = stats_df.columns.astype(str).tolist()
        self.table.setHorizontalHeaderLabels(headers)

        # 填充数据
        for row in range(rows):
            for col in range(cols):
                value = stats_df.iloc[row, col]

                # 格式化数值：浮点数保留2位小数，整数保持原样
                if isinstance(value, float):
                    # 如果是整数形式的浮点数（如 10.0），显示为整数
                    if value == int(value):
                        display_value = str(int(value))
                    else:
                        display_value = f"{value:.2f}"
                else:
                    display_value = str(value)

                item = QTableWidgetItem(display_value)
                # 设置对齐方式：数值右对齐，文本左对齐
                if isinstance(value, (int, float)):
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                else:
                    item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)

                self.table.setItem(row, col, item)

        # 自动调整列宽
        self.table.resizeColumnsToContents()

        # 设置行号（左侧的行标）
        self.table.setVerticalHeaderLabels([str(i) for i in range(1, rows + 1)])

    def _export_statistics(self):
        """导出统计数据表"""
        if not self.analyzer.has_data():
            QMessageBox.warning(self, "警告", "没有数据可导出！")
            return

        stats_df = self.analyzer.get_statistics()
        if stats_df.empty:
            QMessageBox.warning(self, "警告", "没有数值列可供导出统计信息！")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出统计表", "",
            "Excel (*.xlsx);;CSV (*.csv)"
        )

        if file_path:
            success, msg = self.analyzer.export_dataframe(stats_df, file_path)
            if success:
                QMessageBox.information(self, "导出成功", msg)
            else:
                QMessageBox.critical(self, "导出失败", msg)

    def _on_open_file(self):
        if open_file(self, self.analyzer):
            self.refresh()
            self.file_opened.emit()

class PlotPage(QWidget):
    """图像绘制页面"""

    # 定义信号，用于通知主窗口页面切换
    go_to_preview = Signal()
    go_to_statistics = Signal()
    go_to_plot = Signal()
    go_to_advance = Signal()
    file_opened = Signal()

    def __init__(self, analyzer):
        super().__init__()
        self.analyzer = analyzer
        self.selected_x = []  # 存储选中的X列
        self.selected_y = []  # 存储选中的Y列
        self.color_selected = None  # 存储选中的颜色标尺列
        self._setup_ui()
        self._current_fig = None  # 存储当前生成的图像对象

    def _setup_ui(self):
        """初始化UI布局"""
        # 使用水平分割器，左侧放操作按钮，右侧放列选择和绘图按钮
        main_layout = QVBoxLayout(self)

        splitter = QSplitter()
        splitter.setStyleSheet("""
            QSplitter::handle {
                background: #ccc;
                height: 6px;
            }
        """)

        # ===== 左侧面板：操作按钮 =====
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # 绘图按钮
        btn_generate_line = QPushButton("生成折线图")
        btn_generate_line.clicked.connect(lambda: self._generate_plot("line"))

        btn_generate_scatter = QPushButton("生成散点图")
        btn_generate_scatter.clicked.connect(lambda: self._generate_plot("scatter"))

        btn_generate_bar = QPushButton("生成柱状图")
        btn_generate_bar.clicked.connect(lambda: self._generate_plot("bar"))

        btn_export_fig = QPushButton("导出图像")
        btn_export_fig.clicked.connect(self._export_current_figure)

        left_layout.addStretch(1)
        left_layout.addWidget(btn_generate_line)
        left_layout.addWidget(btn_generate_scatter)
        left_layout.addWidget(btn_generate_bar)
        left_layout.addWidget(btn_export_fig)

        # ===== 右侧面板：列选择 + 顶部导航 =====
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # 顶部工具栏
        toolbar = get_toolbar(self)

        # 列选择区域（使用滚动区域防止列太多超出屏幕）
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # X列选择
        x_tip_label = QLabel("选择X列（点击选中/取消，可多选）")
        x_tip_label.setStyleSheet("font-weight: bold; color: blue;")
        self.x_tip_label = x_tip_label

        self.x_grid_layout = QGridLayout()

        # Y列选择
        y_tip_label = QLabel("选择Y列（点击选中/取消，可多选）")
        y_tip_label.setStyleSheet("font-weight: bold; color: red;")
        self.y_tip_label = y_tip_label

        self.y_grid_layout = QGridLayout()

        # 颜色标尺选择
        color_tip_label = QLabel("选择颜色标尺（单选，再次点击取消）")
        color_tip_label.setStyleSheet("font-weight: bold; color: green;")
        self.color_tip_label = color_tip_label

        self.color_grid_layout = QGridLayout()

        scroll_layout.addWidget(x_tip_label)
        scroll_layout.addLayout(self.x_grid_layout)
        scroll_layout.addWidget(y_tip_label)
        scroll_layout.addLayout(self.y_grid_layout)
        scroll_layout.addWidget(color_tip_label)
        scroll_layout.addLayout(self.color_grid_layout)
        scroll_layout.addStretch()

        scroll_area.setWidget(scroll_content)

        right_layout.addLayout(toolbar)
        right_layout.addWidget(scroll_area)

        # 组装分割器
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([200, 500])

        main_layout.addWidget(splitter)

    def refresh(self):
        """刷新页面：重新生成列选择按钮"""
        if not self.analyzer.has_data():
            return

        # 清空之前的选择
        self.selected_x.clear()
        self.selected_y.clear()
        self.color_selected = None

        # 清除旧的按钮
        self._clear_grid_layout(self.x_grid_layout)
        self._clear_grid_layout(self.y_grid_layout)
        self._clear_grid_layout(self.color_grid_layout)

        # 重置提示文字
        self.x_tip_label.setText("选择X列（点击选中/取消，可多选）")
        self.y_tip_label.setText("选择Y列（点击选中/取消，可多选）")
        self.color_tip_label.setText("选择颜色标尺（单选，再次点击取消）")

        # 获取数值列
        numeric_df = self.analyzer.df.select_dtypes(include=[np.number])
        cols = numeric_df.columns.tolist()
        cols_per_row = BTN_FOR_CHOOSE_IN_LINE

        # 创建X列按钮
        for i, col in enumerate(cols):
            btn_x = QPushButton(col)
            btn_x.setCheckable(True)  # 可选中状态

            # 使用默认参数捕获col的当前值，解决闭包问题
            def make_x_action(c=col):
                def action():
                    if c not in self.selected_x:
                        self.selected_x.append(c)
                        self.x_tip_label.setText(f"已选择的X列: {', '.join(self.selected_x)}")
                    else:
                        self.selected_x.remove(c)
                        self.x_tip_label.setText(
                            f"已选择的X列: {', '.join(self.selected_x) if self.selected_x else '无'}")

                return action

            btn_x.clicked.connect(make_x_action())

            row = i // cols_per_row
            col_pos = i % cols_per_row
            self.x_grid_layout.addWidget(btn_x, row, col_pos)

        # 创建Y列按钮
        for i, col in enumerate(cols):
            btn_y = QPushButton(col)
            btn_y.setCheckable(True)

            def make_y_action(c=col):
                def action():
                    if c not in self.selected_y:
                        self.selected_y.append(c)
                        self.y_tip_label.setText(f"已选择的Y列: {', '.join(self.selected_y)}")
                    else:
                        self.selected_y.remove(c)
                        self.y_tip_label.setText(
                            f"已选择的Y列: {', '.join(self.selected_y) if self.selected_y else '无'}")

                return action

            btn_y.clicked.connect(make_y_action())

            row = i // cols_per_row
            col_pos = i % cols_per_row
            self.y_grid_layout.addWidget(btn_y, row, col_pos)

        # 创建颜色标尺按钮
        for i, col in enumerate(cols):
            btn_color = QPushButton(col)
            btn_color.setCheckable(True)

            def make_color_action(c=col):
                def action():
                    if self.color_selected == c:
                        self.color_selected = None
                        self.color_tip_label.setText("选择颜色标尺（单选，再次点击取消）")
                    else:
                        self.color_selected = c
                        self.color_tip_label.setText(f"颜色标尺：{c}")

                return action

            btn_color.clicked.connect(make_color_action())

            row = i // cols_per_row
            col_pos = i % cols_per_row
            self.color_grid_layout.addWidget(btn_color, row, col_pos)

    def _clear_grid_layout(self, grid_layout):
        """清空网格布局中的所有控件"""
        while grid_layout.count():
            item = grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _generate_plot(self, kind="line"):
        """生成图表"""
        if not self.selected_x or not self.selected_y:
            QMessageBox.warning(self, "警告", "请先选择X列和Y列！")
            return

        if not self.analyzer.has_data():
            QMessageBox.warning(self, "警告", "请先加载数据文件！")
            return

        try:
            m = len(self.selected_y)
            n = len(self.selected_x)

            # 创建子图
            fig, axes = plt.subplots(
                m, n,
                figsize=(10, 6),
                sharex=False, sharey=False
            )

            # 统一处理 axes 为二维数组
            if m == 1 and n == 1:
                axes = np.array([[axes]])
            elif m == 1:
                axes = np.array([axes])
            elif n == 1:
                axes = np.array([[ax] for ax in axes])

            df = self.analyzer.df

            for i, y_col in enumerate(self.selected_y):
                for j, x_col in enumerate(self.selected_x):
                    ax = axes[i][j]

                    # 检查数据是否存在
                    if x_col not in df.columns or y_col not in df.columns:
                        ax.text(0.5, 0.5, f"列不存在:\n{x_col} 或 {y_col}",
                                ha='center', va='center')
                        continue

                    x_data = df[x_col]
                    y_data = df[y_col]

                    # 根据图表类型绘图
                    if kind == "line":
                        # 按X排序使折线图更清晰
                        sorted_idx = x_data.argsort()
                        ax.plot(x_data.iloc[sorted_idx], y_data.iloc[sorted_idx],
                                marker="o", linestyle='-', markersize=4)

                    elif kind == "scatter":
                        if self.color_selected and self.color_selected in df.columns:
                            colors = self.get_color(df[self.color_selected])
                            sc = ax.scatter(x_data, y_data, c=colors, s=30, alpha=0.7)
                            # 添加颜色条
                            cbar = plt.colorbar(sc, ax=ax)
                            cbar.set_label(self.color_selected)
                        else:
                            ax.scatter(x_data, y_data, s=30, alpha=0.7)

                    elif kind == "bar":
                        # 柱状图需要处理分类轴
                        if pd.api.types.is_numeric_dtype(x_data):
                            ax.bar(x_data, y_data, width=0.8)
                        else:
                            # 如果X是文本，使用索引作为位置
                            x_pos = range(len(x_data))
                            ax.bar(x_pos, y_data, width=0.8)
                            ax.set_xticks(x_pos)
                            ax.set_xticklabels(x_data, rotation=45, ha='right')

                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    ax.set_title(f"{y_col} vs {x_col}")
                    ax.grid(True, alpha=0.3)
            self._current_fig = fig  # 保存当前图像
            plt.tight_layout()
            plt.show()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"生成图表时出错：\n{str(e)}")

    def _export_current_figure(self):
        """导出当前显示的图像"""
        if self._current_fig is None:
            QMessageBox.warning(self, "警告", "请先生成图像后再导出！")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出图像", "",
            "PNG (*.png);;PDF (*.pdf);;JPG (*.jpg);;SVG (*.svg)"
        )

        if file_path:
            success, msg = self.analyzer.export_figure(self._current_fig, file_path)
            if success:
                QMessageBox.information(self, "导出成功", msg)
            else:
                QMessageBox.critical(self, "导出失败", msg)

    def _on_open_file(self):
        if open_file(self, self.analyzer):
            self.refresh()
            self.file_opened.emit()

class AdvancePage(QWidget):
    """高级统计方法页面"""

    # 定义信号，用于通知主窗口页面切换
    go_to_preview = Signal()
    go_to_statistics = Signal()
    go_to_plot = Signal()
    go_to_advance = Signal()
    file_opened = Signal()

    def __init__(self, analyzer):
        super().__init__()
        self.analyzer = analyzer
        self.selected_x = []  # 存储选中的X列
        self.selected_y = []  # 存储选中的Y列
        self._setup_ui()

    def _setup_ui(self):
        """初始化UI布局"""
        main_layout = QVBoxLayout(self)

        # 使用QStackedWidget管理三个子页面：列选择、帮助、结果
        self.stack = QStackedWidget()

        # 页面0：列选择页面
        self.selection_page = self._create_selection_page()
        # 页面1：帮助页面
        self.help_page = self._create_help_page()
        # 页面2：结果页面
        self.result_page = self._create_result_page()

        self.stack.addWidget(self.selection_page)  # index 0
        self.stack.addWidget(self.help_page)  # index 1
        self.stack.addWidget(self.result_page)  # index 2

        main_layout.addWidget(self.stack)

    def _create_selection_page(self):
        """创建列选择页面（包含导航栏、列选择和操作按钮）"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # 使用分割器
        splitter = QSplitter()
        splitter.setStyleSheet("""
            QSplitter::handle {
                background: #ccc;
                height: 6px;
            }
        """)

        # ===== 左侧面板：操作按钮 =====
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)



        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)

        btn_help = QPushButton("帮助")
        btn_help.clicked.connect(self._show_help)

        btn_linear = QPushButton("线性回归方法")
        btn_linear.clicked.connect(lambda: self._run_analysis("linear_regression"))

        btn_forest = QPushButton("随机森林方法")
        btn_forest.clicked.connect(lambda: self._run_analysis("random_forest"))

        left_layout.addStretch(1)
        left_layout.addWidget(line)
        left_layout.addWidget(btn_help)
        left_layout.addWidget(btn_linear)
        left_layout.addWidget(btn_forest)

        # ===== 右侧面板：列选择 =====
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # 顶部工具栏
        toolbar = get_toolbar(self)

        # 列选择区域（使用滚动区域）
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # X列选择
        x_tip_label = QLabel("选择自变量X（点击选中/取消，可多选）")
        x_tip_label.setStyleSheet("font-weight: bold; color: blue;")
        self.x_tip_label = x_tip_label

        self.x_grid_layout = QGridLayout()

        # Y列选择
        y_tip_label = QLabel("选择因变量Y（点击选中/取消，只能选一个）")
        y_tip_label.setStyleSheet("font-weight: bold; color: red;")
        self.y_tip_label = y_tip_label

        self.y_grid_layout = QGridLayout()

        scroll_layout.addWidget(x_tip_label)
        scroll_layout.addLayout(self.x_grid_layout)
        scroll_layout.addWidget(y_tip_label)
        scroll_layout.addLayout(self.y_grid_layout)
        scroll_layout.addStretch()

        scroll_area.setWidget(scroll_content)

        right_layout.addLayout(toolbar)
        right_layout.addWidget(scroll_area)

        # 组装分割器
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([200, 700])

        layout.addWidget(splitter)
        return widget

    def _create_help_page(self):
        """创建帮助页面"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # 顶部导航
        nav_bar = QHBoxLayout()
        btn_back = QPushButton("← 返回选择页面")
        btn_back.clicked.connect(lambda: self.stack.setCurrentIndex(0))

        btn_stats = QPushButton("浏览统计数据")
        btn_stats.clicked.connect(self.go_to_statistics.emit)

        btn_plot = QPushButton("图像绘制")
        btn_plot.clicked.connect(self.go_to_plot.emit)

        nav_bar.addWidget(btn_back)
        nav_bar.addStretch()
        nav_bar.addWidget(btn_stats)
        nav_bar.addWidget(btn_plot)

        # 帮助文本
        help_text = """
        ============================================================
                        高级统计方法 - 帮助文档
        ============================================================

        【线性回归】
        线性回归用于研究一个或多个自变量(X)与因变量(Y)之间的线性关系。

        参数含义：
        ────────────────────────────────────────────
        Dep. Variable      ：被解释变量（Y）
        Method             ：最小二乘法（Least Squares）
        No. Observations   ：样本数量
        Df Model           ：自变量个数
        R-squared          ：决定系数（0~1），表示模型解释力
        Adj. R-squared     ：调整后的R²，更稳健
        F-statistic        ：F统计量，检验模型整体显著性
        Prob (F-statistic) ：F检验的p值（<0.05表示显著）
        Log-Likelihood     ：对数似然值
        const              ：截距项
        coef               ：回归系数（斜率）
        std err            ：系数的标准误
        P>|t|              ：系数的p值（<0.05表示显著）
        Omnibus            ：残差正态性检验
        Durbin-Watson      ：残差自相关检验（接近2为佳）
        Jarque-Bera (JB)   ：残差正态性检验
        Cond. No.          ：多重共线性诊断（<100较安全）

        【随机森林】
        随机森林是一种集成学习方法，通过构建多棵决策树并取平均值来提高预测精度。

        输出指标含义：
        ────────────────────────────────────────────
        R² (R-squared)     ：决定系数，越接近1越好
        MSE                ：均方误差，越小越好
        RMSE               ：均方根误差，越小越好
        MAE                ：平均绝对误差，越小越好
        特征重要性         ：各特征对预测的贡献度

        注意事项：
        ────────────────────────────────────────────
        1. 请确保数据中没有缺失值（NaN）
        2. 分类变量需要先转换为数值型
        3. 建议至少选择2个以上样本
        4. 训练/测试比例为70%/30%
        """

        # 滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        text_label = QLabel()
        text_label.setWordWrap(True)
        text_label.setTextFormat(Qt.PlainText)
        text_label.setFont(QFont("Microsoft YaHei", 10))
        text_label.setText(help_text)

        scroll.setWidget(text_label)

        layout.addLayout(nav_bar)
        layout.addWidget(scroll)
        return widget

    def _create_result_page(self):
        """创建结果展示页面"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # 顶部导航
        nav_bar = QHBoxLayout()
        btn_back = QPushButton("← 返回选择页面")
        btn_back.clicked.connect(lambda: self.stack.setCurrentIndex(0))

        btn_help = QPushButton("帮助")
        btn_help.clicked.connect(self._show_help)

        btn_stats = QPushButton("浏览统计数据")
        btn_stats.clicked.connect(self.go_to_statistics.emit)

        btn_plot = QPushButton("图像绘制")
        btn_plot.clicked.connect(self.go_to_plot.emit)

        btn_export_result = QPushButton("导出结果")
        btn_export_result.clicked.connect(self._export_result)

        nav_bar.addWidget(btn_back)
        nav_bar.addStretch()
        nav_bar.addWidget(btn_help)
        nav_bar.addWidget(btn_stats)
        nav_bar.addWidget(btn_plot)
        nav_bar.addWidget(btn_export_result)

        # 结果文本编辑器
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setFont(QFont("Consolas", 10))  # 等宽字体显示表格

        layout.addLayout(nav_bar)
        layout.addWidget(self.result_text)
        return widget

    def refresh(self):
        """刷新页面：重新生成列选择按钮"""
        if not self.analyzer.has_data():
            return

        # 清空之前的选择
        self.selected_x.clear()
        self.selected_y.clear()

        # 清除旧的按钮
        self._clear_grid_layout(self.x_grid_layout)
        self._clear_grid_layout(self.y_grid_layout)

        # 重置提示文字
        self.x_tip_label.setText("选择自变量X（点击选中/取消，可多选）")
        self.y_tip_label.setText("选择因变量Y（点击选中/取消，只能选一个）")

        # 获取数值列
        numeric_df = self.analyzer.df.select_dtypes(include=[np.number])
        cols = numeric_df.columns.tolist()
        cols_per_row = BTN_FOR_CHOOSE_IN_LINE

        # 创建X列按钮（可多选）
        for i, col in enumerate(cols):
            btn_x = QPushButton(col)
            btn_x.setCheckable(True)

            def make_x_action(c=col):
                def action():
                    if c not in self.selected_x:
                        self.selected_x.append(c)
                        self.x_tip_label.setText(
                            f"已选择的X列: {', '.join(self.selected_x)}"
                        )
                    else:
                        self.selected_x.remove(c)
                        self.x_tip_label.setText(
                            f"已选择的X列: {', '.join(self.selected_x) if self.selected_x else '无'}"
                        )

                return action

            btn_x.clicked.connect(make_x_action())

            row = i // cols_per_row
            col_pos = i % cols_per_row
            self.x_grid_layout.addWidget(btn_x, row, col_pos)

        # 创建Y列按钮（只能单选）
        for i, col in enumerate(cols):
            btn_y = QPushButton(col)
            btn_y.setCheckable(True)

            def make_y_action(c=col):
                def action():
                    if c not in self.selected_y:
                        # 清空之前的Y选择
                        self.selected_y.clear()
                        self.selected_y.append(c)
                        self.y_tip_label.setText(f"已选择的Y列: {c}")
                    else:
                        self.selected_y.remove(c)
                        self.y_tip_label.setText("选择因变量Y（点击选中/取消，只能选一个）")

                return action

            btn_y.clicked.connect(make_y_action())

            row = i // cols_per_row
            col_pos = i % cols_per_row
            self.y_grid_layout.addWidget(btn_y, row, col_pos)

    def _clear_grid_layout(self, grid_layout):
        """清空网格布局中的所有控件"""
        while grid_layout.count():
            item = grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _show_help(self):
        """显示帮助页面"""
        self.stack.setCurrentIndex(1)

    def _run_analysis(self, method="linear_regression"):
        """执行统计分析"""
        if not self.selected_x or not self.selected_y:
            QMessageBox.warning(self, "警告", "请先选择自变量X和因变量Y！")
            return

        if not self.analyzer.has_data():
            QMessageBox.warning(self, "警告", "请先加载数据文件！")
            return

        try:
            # 执行分析
            result = self._advance_statistics(method)

            # 显示结果
            self.result_text.setPlainText(result)
            self.stack.setCurrentIndex(2)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"分析过程中出错：\n{str(e)}")

    def _advance_statistics(self, method="linear_regression"):
        """执行统计计算，返回结果字符串"""
        if method == "linear_regression":
            return self._linear_regression()
        elif method == "random_forest":
            return self._random_forest()
        return "未知的分析方法"

    def _linear_regression(self):
        """执行线性回归"""
        df = self.analyzer.df
        x = df[self.selected_x]
        y = df[self.selected_y[0]]

        # 去除缺失值
        valid_idx = x.dropna().index.intersection(y.dropna().index)
        x = x.loc[valid_idx]
        y = y.loc[valid_idx]

        X = sm.add_constant(x)
        model = sm.OLS(y, X)
        results = model.fit()

        # 格式化输出
        output = [
            "=" * 72,
            "                    线性回归分析报告",
            "=" * 72,
            "",
            f"因变量 (Y): {self.selected_y[0]}",
            f"自变量 (X): {', '.join(self.selected_x)}",
            f"有效样本数: {len(valid_idx)}",
            "",
            "-" * 40,
            "回归结果摘要",
            "-" * 40,
        ]
        output.append(str(results.summary()))

        return "\n".join(output)

    def _random_forest(self):
        """执行随机森林回归"""
        df = self.analyzer.df
        X = df[self.selected_x].dropna()
        y = df[self.selected_y[0]].loc[X.index]

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # 训练模型
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            random_state=42
        )
        model.fit(X_train, y_train)

        # 预测
        y_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)

        # 计算指标
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_test, y_pred)

        train_r2 = r2_score(y_train, y_train_pred)
        train_rmse = mean_squared_error(y_train, y_train_pred) ** 0.5

        # 构建结果文本
        result_lines = [
            "=" * 68,
            "              随机森林回归模型评估报告",
            "=" * 68,
            "",
            f"因变量 (Y): {self.selected_y[0]}",
            f"自变量 (X): {', '.join(self.selected_x)}",
            "",
            "┌─────────────────────────────────────────────────────┐",
            "│                    数据概况                         │",
            "├─────────────────────────────────────────────────────┤",
            f"│  总样本数         : {len(X):>15d}                  │",
            f"│  训练集样本数     : {len(X_train):>15d}            │",
            f"│  测试集样本数     : {len(X_test):>15d}             │",
            f"│  特征数           : {X.shape[1]:>15d}              │",
            f"│  测试集比例       : {'30%':>16s}                   │",
            "└─────────────────────────────────────────────────────┘",
            "",
            "┌─────────────────────────────────────────────────────┐",
            "│                    模型参数                         │",
            "├─────────────────────────────────────────────────────┤",
            "│  模型类型         : RandomForestRegressor           │",
            f"│  树的数量         : {'100':>16s}                    │",
            f"│  最大深度         : {'None (不限制)':>16s}          │",
            "└─────────────────────────────────────────────────────┘",
            "",
            "┌─────────────────────────────────────────────────────┐",
            "│                  测试集性能                         │",
            "├─────────────────────────────────────────────────────┤",
            f"│  R² 决定系数      : {r2:>15.4f}                    │",
            f"│  MSE 均方误差     : {mse:>15.4f}                   │",
            f"│  RMSE 均方根误差  : {rmse:>15.4f}                  │",
            f"│  MAE 平均绝对误差 : {mae:>15.4f}                   │",
            "└─────────────────────────────────────────────────────┘",
            "",
            "┌─────────────────────────────────────────────────────┐",
            "│                  训练集性能                         │",
            "├─────────────────────────────────────────────────────┤",
            f"│  R²               : {train_r2:>15.4f}              │",
            f"│  RMSE             : {train_rmse:>15.4f}            │",
            "└─────────────────────────────────────────────────────┘",
            "",
            "┌─────────────────────────────────────────────────────┐",
            "│                  过拟合判断                         │",
            "├─────────────────────────────────────────────────────┤",
            f"│  训练/测试 R²差   : {abs(train_r2 - r2):>13.4f}    │",
            f"│  结论             : {'可能存在过拟合' if train_r2 - r2 > 0.1 else '未见明显过拟合':>14s}│",
            "└─────────────────────────────────────────────────────┘",
            "",
            "┌─────────────────────────────────────────────────────┐",
            "│                  特征重要性                         │",
            "├─────────────────────────────────────────────────────┤",
        ]

        # 添加特征重要性
        for col, imp in sorted(
                zip(self.selected_x, model.feature_importances_),
                key=lambda x: x[1],
                reverse=True
        ):
            bar_length = int(imp * 50)
            bar = "█" * bar_length
            result_lines.append(f"│  {col:<15s} : {imp:.4f}  {bar:<50s}│")

        result_lines.append("└─────────────────────────────────────────────────────┘")

        return "\n".join(result_lines)

    def _export_result(self):
        """导出分析结果"""
        current_text = self.result_text.toPlainText()
        if not current_text.strip():
            QMessageBox.warning(self, "警告", "没有分析结果可导出！请先运行分析方法。")
            return

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "导出分析结果", "",
            "文本文件 (*.txt);;Excel 文件 (*.xlsx)"
        )

        if not file_path:
            return

        # 判断选择的格式
        if selected_filter == "Excel 文件 (*.xlsx)" or file_path.endswith('.xlsx'):
            # 尝试将文本解析为结构化表格导出
            lines = current_text.split('\n')
            records = []
            for line in lines:
                line = line.strip()
                if ':' in line and not line.startswith('=') and not line.startswith('┌') \
                        and not line.startswith('├') and not line.startswith('└') \
                        and not line.startswith('│') and not line.startswith('-') \
                        and not line.startswith('*'):
                    parts = line.split(':', 1)
                    key = parts[0].strip()
                    value = parts[1].strip() if len(parts) > 1 else ''
                    records.append({"指标": key, "值": value})

            if records:
                result_df = pd.DataFrame(records)
                success, msg = self.analyzer.export_dataframe(result_df, file_path)
            else:
                # 无法解析，降级为文本导出
                if not file_path.endswith('.txt'):
                    file_path += '.txt'
                success, msg = self.analyzer.export_text(current_text, file_path)
        else:
            success, msg = self.analyzer.export_text(current_text, file_path)

        if success:
            QMessageBox.information(self, "导出成功", msg)
        else:
            QMessageBox.critical(self, "导出失败", msg)

    def _on_open_file(self):
        if open_file(self, self.analyzer):
            self.refresh()
            self.file_opened.emit()


class MainWindow(QMainWindow):
    """主窗口，负责页面切换和协调"""

    def __init__(self):
        super().__init__()
        self.analyzer = DataAnalyzer()
        self._setup_ui()

        # 恢复窗口设置
        self._restore_window_settings()

    def _restore_window_settings(self):
        """从配置文件恢复窗口大小和位置"""
        config = load_config()

        width = config.get("window_width", 1200)
        height = config.get("window_height", 800)
        x = config.get("window_x")
        y = config.get("window_y")

        self.resize(width, height)

        if x is not None and y is not None:
            # 检查屏幕边界，防止窗口移出屏幕
            screen = QApplication.primaryScreen().geometry()
            if 0 <= x <= screen.width() - 100 and 0 <= y <= screen.height() - 100:
                self.move(x, y)

        # 如果有上次的文件路径且文件存在，自动加载
        last_path = config.get("last_file_path", "")
        if last_path and os.path.exists(last_path):
            if self.analyzer.load_file(last_path):
                # 刷新预览页面
                self.preview_page.refresh()

    def closeEvent(self, event):
        """重写关闭事件，保存窗口设置"""
        config = load_config()

        # 保存窗口大小和位置
        config["window_width"] = self.width()
        config["window_height"] = self.height()
        config["window_x"] = self.x()
        config["window_y"] = self.y()

        save_config(config)
        #窗口关闭时，自动保存状态
        super().closeEvent(event)

    def _setup_ui(self):
        self.setWindowTitle("数据处理软件")
        self.resize(1200, 800)

        # 创建中心部件和堆叠布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.stacked_widget = QStackedWidget()

        # 创建各个页面
        self.preview_page = PreviewPage(self.analyzer)
        self.statistics_page = StatisticsPage(self.analyzer)
        self.plot_page = PlotPage(self.analyzer)
        self.advance_page = AdvancePage(self.analyzer)

        # 添加到堆叠布局
        self.stacked_widget.addWidget(self.preview_page)  # index 0
        self.stacked_widget.addWidget(self.statistics_page)  # index 1
        self.stacked_widget.addWidget(self.plot_page)  # index 2
        self.stacked_widget.addWidget(self.advance_page)  # index 3

        # 连接信号
        self._connect_signals()

        # 主布局
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.stacked_widget)

        # 默认显示预览页
        self.stacked_widget.setCurrentIndex(0)

    def _connect_signals(self):
        """连接所有页面的信号到主窗口的槽函数"""

        # 预览页
        self.preview_page.go_to_statistics.connect(
            lambda: self.switch_to_page(1)
        )
        self.preview_page.go_to_plot.connect(
            lambda: self.switch_to_page(2)
        )
        self.preview_page.go_to_advance.connect(
            lambda: self.switch_to_page(3)
        )
        self.preview_page.file_opened.connect(
            lambda: self.switch_to_page(0)
        )

        # 统计页
        self.statistics_page.go_to_preview.connect(
            lambda: self.switch_to_page(0)
        )
        self.statistics_page.go_to_plot.connect(
            lambda: self.switch_to_page(2)
        )
        self.statistics_page.go_to_advance.connect(
            lambda: self.switch_to_page(3)
        )

        # 绘图页
        self.plot_page.go_to_preview.connect(
            lambda: self.switch_to_page(0)
        )
        self.plot_page.go_to_statistics.connect(
            lambda: self.switch_to_page(1)
        )
        self.plot_page.go_to_advance.connect(
            lambda: self.switch_to_page(3)
        )

        # 高级统计页
        self.advance_page.go_to_preview.connect(
            lambda: self.switch_to_page(0)
        )
        self.advance_page.go_to_statistics.connect(
            lambda: self.switch_to_page(1)
        )
        self.advance_page.go_to_plot.connect(
            lambda: self.switch_to_page(2)
        )

    def switch_to_page(self, index: int):
        """切换到指定页面并刷新"""
        page = self.stacked_widget.widget(index)
        if hasattr(page, 'refresh'):
            page.refresh()
        self.stacked_widget.setCurrentIndex(index)


def main():
    app = QApplication(sys.argv)

    # 全局样式
    app.setStyle('Fusion')

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()