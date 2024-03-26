import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 载入数据
df = pd.read_csv('Student Performance Dataset.csv')

# 创建用于保存图形的文件夹
plot_folder = 'matplot'
if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)

# 定义保存图形的函数
def save_fig(fig, filename):
    fig.savefig(os.path.join(plot_folder, filename))

# 学生的整体成绩分布
sns.histplot(df['G3'], bins=20, kde=True)
plt.title('学生最终成绩分布')
plt.xlabel('成绩')
plt.ylabel('学生人数')
save_fig(plt, '成绩分布.png')
plt.clf()

# 性别对成绩的影响
sns.boxplot(x='sex', y='G3', data=df)
plt.title('性别对最终成绩的影响')
plt.xlabel('性别')
plt.ylabel('成绩')
save_fig(plt, '性别对成绩的影响.png')
plt.clf()

# 出勤率与成绩的关系
sns.scatterplot(x='absences', y='G3', data=df)
plt.title('出勤率与最终成绩的关系')
plt.xlabel('缺席次数')
plt.ylabel('最终成绩')
save_fig(plt, '出勤率与成绩的关系.png')
plt.clf()

# 家长教育水平对学生成绩的影响
df['MaxParentEdu'] = df[['Medu', 'Fedu']].max(axis=1)
sns.boxplot(x='MaxParentEdu', y='G3', data=df)
plt.title('家长教育水平对学生成绩的影响')
plt.xlabel('家长最高教育水平')
plt.ylabel('最终成绩')
save_fig(plt, '家长教育水平对成绩的影响.png')
plt.clf()

# 科目成绩与最终成绩的相关性
correlation = df[['G1', 'G2', 'G3']].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('科目成绩与最终成绩的相关性')
save_fig(plt, '科目成绩与最终成绩的相关性.png')
plt.clf()

# 参与课外活动与成绩的关系
sns.barplot(x='activities', y='G3', data=df)
plt.title('参与课外活动与最终成绩的关系')
plt.xlabel('是否参与课外活动')
plt.ylabel('平均成绩')
save_fig(plt, '参与课外活动与成绩的关系.png')
plt.clf()

# 学习时间与成绩的关系
sns.scatterplot(x='studytime', y='G3', data=df)
plt.title('学习时间与最终成绩的关系')
plt.xlabel('每周学习时间（小时）')
plt.ylabel('最终成绩')
save_fig(plt, '学习时间与成绩的关系.png')
plt.clf()

# 网络资源使用与成绩的关系
sns.boxplot(x='internet', y='G3', data=df)
plt.title('网络资源使用与最终成绩的关系')
plt.xlabel('是否使用网络资源')
plt.ylabel('最终成绩')
save_fig(plt, '网络资源使用与成绩的关系.png')
plt.clf()

# 不同年龄对成绩的影响
sns.boxplot(x='age', y='G3', data=df)
plt.title('不同年龄（学年级）对成绩的影响')
plt.xlabel('年龄')
plt.ylabel('最终成绩')
save_fig(plt, '不同年龄对成绩的影响.png')
plt.clf()
