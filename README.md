# Aviation Data Analysis

Aviation_Data is a class responsible for handling AviationData dataset, creating stats and visualization.

## Requirements

The *conda* virtual environment was used for this project.

To use it you need to install following libraries:

- pandas
- seaborn
- sklearn 

or refer to *requirements.txt*. 

The dataset used is [AviationData](https://www.kaggle.com/datasets/khsamaha/aviation-accident-database-synopses/data).

To execute one of the methods - *heatmap_locations()* - you also need *coordinates_data.csv*. You can find it on [NTSB official page](https://www.ntsb.gov/safety/data/Pages/Data_Stats.aspx).

## Usage

You need to create object of Aviation_Data class in *data_analysis_main.py* and execute methods from the main.

```python
# creates Aviation_Data object
data = Aviation_Data()

# saves basic stats to csv files
basic_stats(data)

# creates plots for different correlations between data
# After closing one plot, another will appear
show_all_plots(data)
```

These two methods are executed **by default**. 

## Author

Maja Kurczyna (280494)