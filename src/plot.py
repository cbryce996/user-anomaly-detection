# Internal imports
import util as utl

# External imports
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid')

def plot_heatmap(data, file_name, title, size):
    plt.clf()
    plt.figure(figsize=size)
    sns.heatmap(data)
    plt.title(title)
    plt.tight_layout()
    plt.savefig('../assets/images/heat_map_%s.png' % (utl.file_string(file_name)))

def plot_bar(data, file_name, title, size, xlabel, ylabel):
    plt.clf()
    plt.figure(figsize=size)
    sns.barplot(x='X', y='Y', data=data, palette=sns.color_palette('viridis', n_colors=45))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig('../assets/images/bar_plot_%s.png' % (utl.file_string(file_name)))

def plot_line(data, file_name, title, size, xlabel, ylabel, score=''):
    plt.clf()
    plt.figure(figsize=size)
    for name, values in data.items():
        if not score:
             sns.lineplot(data=data[name], x='X', y='Y')
        else:
            label = values['Label']
            sns.lineplot(data=data[name], x='X', y='Y', label='%s - %s: %s' % (name, score,label))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.legend()
    plt.savefig('../assets/images/line_plot_%s.png' % (utl.file_string(file_name)))