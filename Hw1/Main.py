import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.font_manager import FontProperties

# 設置Matplotlib的字體
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 例如，使用"Microsoft YaHei"字體
plt.rcParams['axes.unicode_minus'] = False  # 解決負號（-）顯示為方塊的問題

# 載入數據
df = pd.read_csv('Student Performance Dataset.csv')

# 創建用於保存圖形的文件夾
plot_folder = 'matplot'
if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)  # 如果文件夾不存在，則創建文件夾

# 定義保存圖形的函數
def save_fig(fig, filename):
    fig.savefig(os.path.join(plot_folder, filename))  # 保存圖形到指定路徑

# 學生的整體成績分布
sns.histplot(df['G3'], bins=20, kde=True)
plt.title('學生最終成績分布')
plt.xlabel('成績')
plt.ylabel('學生人數')
save_fig(plt, '成績分布.png')  # 調用save_fig函數保存圖片
plt.clf()  # 清除當前圖形

# 性別對成績的影響
sns.boxplot(x='sex', y='G3', data=df)
plt.title('性別對最終成績的影響')
plt.xlabel('性別')
plt.ylabel('成績')
save_fig(plt, '性別對成績的影響.png')  # 調用save_fig函數保存圖片
plt.clf()

# 出勤率與成績的關係
sns.scatterplot(x='absences', y='G3', data=df)
plt.title('出勤率與最終成績的關係')
plt.xlabel('缺席次數')
plt.ylabel('最終成績')
save_fig(plt, '出勤率與成績的關係.png')  # 調用save_fig函數保存圖片
plt.clf()

# 家長教育水平對學生成績的影響
df['MaxParentEdu'] = df[['Medu', 'Fedu']].max(axis=1)  # 計算家長的最高教育水平
sns.boxplot(x='MaxParentEdu', y='G3', data=df)
plt.title('家長教育水平對學生成績的影響')
plt.xlabel('家長最高教育水平')
plt.ylabel('最終成績')
save_fig(plt, '家長教育水平對成績的影響.png')  # 調用save_fig函數保存圖片
plt.clf()

# 科目成績與最終成績的相關性
correlation = df[['G1', 'G2', 'G3']].corr()  # 計算科目成績間的相關係數
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('科目成績與最終成績的相關性')
save_fig(plt, '科目成績與最終成績的相關性.png')  # 調用save_fig函數保存圖片
plt.clf()

# 參與課外活動與成績的關係
sns.barplot(x='activities', y='G3', data=df)
plt.title('參與課外活動與最終成績的關係')
plt.xlabel('是否參與課外活動')
plt.ylabel('平均成績')
save_fig(plt, '參與課外活動與成績的關係.png')  # 調用save_fig函數保存圖片
plt.clf()

# 學習時間與成績的關係
sns.scatterplot(x='studytime', y='G3', data=df)
plt.title('學習時間與最終成績的關係')
plt.xlabel('每周學習時間（小時）')
plt.ylabel('最終成績')
save_fig(plt, '學習時間與成績的關係.png')  # 調用save_fig函數保存圖片
plt.clf()

# 網絡資源使用與成績的關係
sns.boxplot(x='internet', y='G3', data=df)
plt.title('網絡資源使用與最終成績的關係')
plt.xlabel('是否使用網絡資源')
plt.ylabel('最終成績')
save_fig(plt, '網絡資源使用與成績的關係.png')  # 調用save_fig函數保存圖片
plt.clf()

# 不同年齡對成績的影響
sns.boxplot(x='age', y='G3', data=df)
plt.title('不同年齡（學年級）對成績的影響')
plt.xlabel('年齡')
plt.ylabel('最終成績')
save_fig(plt, '不同年齡對成績的影響.png')  # 調用save_fig函數保存圖片
plt.clf()
