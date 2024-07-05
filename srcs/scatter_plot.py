from matplotlib import pyplot

from dataset import Dataset
from histogram import categorise_rows

def main():

    target_feature = "Hogwarts House"
    features = ["Arithmancy",
                "Astronomy",
                "Herbology",
                "Defense Against the Dark Arts",
                "Divination",
                "Muggle Studies",
                "Ancient Runes",
                "History of Magic",
                "Transfiguration",
                "Potions",
                "Care of Magical Creatures",
                "Charms",
                "Flying"]

    ds = Dataset("datasets/dataset_train.csv")

    ds.clean(target_feature, features)

    target_data = ds.get_target()
    
    classes = categorise_rows(target_data)
    
    num_features = len(features)
    num_classes = len(classes)
    
    subplot_size = 5
    fig_size = num_features * subplot_size
    fig, axs = pyplot.subplots(num_features, num_features, figsize=(fig_size, fig_size))
    
    plot_color = ["red", "deepskyblue", "springgreen", "gold"]
    
    for x, x_feature in enumerate(features):
        
        for y, y_feature in enumerate(features):
            
            axs[x, y].set_ylabel(x_feature, fontsize=8)
            axs[x, y].set_xlabel(y_feature, fontsize=8)
            
            for i, (house, indices) in enumerate(classes.items()):
                
                x_data = [ds.get_data()[i][x] for i in indices]
                y_data = [ds.get_data()[i][y] for i in indices]

                axs[x, y].scatter(x_data, y_data, color=plot_color[i], alpha=0.5, label=house)
            
            axs[x, y].legend(loc='upper right', fontsize='x-small')

    pyplot.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,
                           wspace=0.5, hspace=0.5)

    pyplot.savefig("graphs/scatter_plot.png")

if __name__ == "__main__":

    main()