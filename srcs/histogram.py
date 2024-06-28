from matplotlib import pyplot

from describe import extract, clean

def get_target_feature(headers: list[str], raw: list[list]):
    
    target_feature = "Hogwarts House"
    
    idx = headers.index(target_feature)
    
    return [row[idx] for row in raw]

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

    headers, extracted = extract("../datasets/dataset_train.csv")

    cleaned_headers, cleaned = clean(headers, extracted)

    target_data = get_target_feature(headers, extracted)
    
    classes = categorise_rows(target_data)
    
    num_features = len(cleaned_headers)
    num_classes = len(classes)
    
    subplot_size = 3
    fig_width = num_features * subplot_size
    fig_height = (num_classes + 1) * subplot_size
    fig, axs = pyplot.subplots(num_classes + 1, num_features, figsize=(fig_width, fig_height))
    
    plot_color = ["red", "deepskyblue", "springgreen", "gold"]
    
    for j, feature in enumerate(cleaned_headers):
        
        for i, (house, indices) in enumerate(classes.items()):
            
            data = [cleaned[k][j] for k in indices]
        
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
    
    pyplot.savefig("../assets/histogram.png")
    
    

if __name__ == "__main__":

    main()