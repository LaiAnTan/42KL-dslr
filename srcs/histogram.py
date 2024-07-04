from matplotlib import pyplot

from dataset import Dataset

def categorise_rows(target_data: list[str]):
    
    classes = {
        "Gryffindor": [],
        "Ravenclaw": [],
        "Slytherin": [],
        "Hufflepuff": []
    }
    
    for i, val in enumerate(target_data):
        
        classes[val].append(i)
    
    return classes

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
    
    subplot_size = 3
    fig_width = num_features * subplot_size
    fig_height = (num_classes + 1) * subplot_size
    fig, axs = pyplot.subplots(num_classes + 1, num_features, figsize=(fig_width, fig_height))
    
    plot_color = ["red", "deepskyblue", "springgreen", "gold"]
    
    for j, feature in enumerate(features):
        
        for i, (house, indices) in enumerate(classes.items()):
            
            data = [ds.get_data()[k][j] for k in indices]
        
            axs[i, j].hist(data, color=plot_color[i])
            axs[i, j].set_title(f"{house} - {feature}", fontsize=8)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])

            axs[4, j].hist(data, color=plot_color[i], alpha=0.5)
        
        axs[4, j].set_title(feature, fontsize=8)
        axs[4, j].set_xticks([])
        axs[4, j].set_yticks([])

    pyplot.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,
                           wspace=0.5, hspace=0.5)
    
    pyplot.savefig("assets/histogram.png")
    
    

if __name__ == "__main__":

    main()