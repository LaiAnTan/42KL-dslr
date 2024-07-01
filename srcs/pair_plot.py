from matplotlib import pyplot

from describe import extract, clean
from histogram import categorise_rows, get_target_feature
from stats import Statistics

def main():

    headers, extracted = extract("../datasets/dataset_train.csv")

    cleaned_headers, cleaned = clean(headers, extracted)

    target_data = get_target_feature(headers, extracted)
    
    classes = categorise_rows(target_data)
    
    num_features = len(cleaned_headers)
    num_classes = len(classes)
    
    subplot_size = 5
    fig_size = num_features * subplot_size
    fig, axs = pyplot.subplots(num_features, num_features, figsize=(fig_size, fig_size))
    
    plot_color = ["red", "deepskyblue", "springgreen", "gold"]
    
    for x, x_feature in enumerate(cleaned_headers):
        
        for y, y_feature in enumerate(cleaned_headers):
            
            axs[x, y].set_ylabel(x_feature, fontsize=8)
            axs[x, y].set_xlabel(y_feature, fontsize=8)
            
            if x == y:
                
                for i, (house, indices) in enumerate(classes.items()):
                    
                    data = [cleaned[k][x] for k in indices]
                    axs[x, y].hist(data, color=plot_color[i], alpha=0.5, label=house)
                
            else:
                
                for i, (house, indices) in enumerate(classes.items()):
                    
                    x_data = [cleaned[i][x] for i in indices]
                    y_data = [cleaned[i][y] for i in indices]
    
                    axs[x, y].scatter(x_data, y_data, color=plot_color[i], alpha=0.5, label=house)
            
            axs[x, y].legend(loc='upper right', fontsize='x-small')

    pyplot.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,
                           wspace=0.5, hspace=0.5)

    pyplot.savefig("../assets/pair_plot.png")

if __name__ == "__main__":

    main()