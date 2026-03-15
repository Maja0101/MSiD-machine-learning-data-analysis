from Aviation_Data import Aviation_Data

def basic_stats(data: Aviation_Data):
    """Saves to .csv files basic numerical statistics, basic categorical statistics
    and values proportions for categorical columns"""

    data.create_basic_numerical_statistics().to_csv('basic_numerical_statistics.csv')
    data.create_basic_categorical_statistics().to_csv('basic_categorical_statistics.csv')

    for column_name in data.categorical_columns.to_list():
        name = 'proportions_' + column_name.replace('.', '_') + '.csv'
        data.create_values_proportions(column_name).to_csv(name)

def show_all_plots(data: Aviation_Data):
    """Shows different plots made based on aviation data"""

    data.boxplot_injuries_percentage()
    data.violinplot_fatalities_percentage()
    data.errorbar_time_to_report()
    data.histoplot_accident_by_phase_of_flight()
    data.histplot_acciedent_month_hue_weather()
    data.heatmap_locations()
    data.regplot_accidents_through_years()
    data.pca()

if __name__ == "__main__":
    data = Aviation_Data()
    # basic_stats(data)
    show_all_plots(data)