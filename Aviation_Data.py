import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from math import isnan
from calendar import month_name
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

class Aviation_Data:
    def __init__(self):
        self.file_name = 'AviationData.csv'
        self.__data_set = None
        self.__read_aviation_data()
        self._lat_long_indexes = [], []
        self.__invalid = -9999

    def __read_aviation_data(self):
        """Create dataframe from .csv file with aviation data"""

        self.__data_set = pd.read_csv(self.file_name, encoding='ISO-8859-1', low_memory=False)
        self.__data_set['Event.Date'] = pd.to_datetime(self.__data_set['Event.Date'], format="%Y-%m-%d")
        self.__data_set['Publication.Date'] = pd.to_datetime(self.__data_set['Publication.Date'], format="%d-%m-%Y")

        #Determine names of numerical and categorical columns
        self.numerical_columns = self.__data_set.select_dtypes(exclude='object').columns 
        self.categorical_columns = self.__data_set.select_dtypes(include='object').columns 

    def get_data_set(self):
        return self.__data_set
        
    def create_basic_numerical_statistics(self):
        """Retruns dataframe with data such as  numebr of non-null values, average, min, max, 5th percentile, median, 95th percentile,
        standard deviation, number of null values for numerical data"""

        basic_info = self.__data_set.describe(percentiles=[0.05, 0.95])
        missing_values = pd.DataFrame([self.__data_set.isna().sum()], columns=self.__data_set.columns)
        missing_values.index = ['NaN']
        stats = pd.concat([basic_info, missing_values[basic_info.columns]], ignore_index=True)
        stats.index = basic_info.index.to_list() + missing_values.index.to_list()
        return stats
    
    def create_basic_categorical_statistics(self):
        """Returns dataframe with data such as number of unique values, number of null values for categorical data"""

        stats = pd.DataFrame([
            self.__data_set[self.categorical_columns].nunique(),
            self.__data_set[self.categorical_columns].isna().sum()
        ], columns=self.categorical_columns)
        stats.index = ['Unique', 'NaN']
        return stats
    
    def create_values_proportions(self, column_name):
        """Returns values proportion for specified column"""

        return self.__data_set[column_name].value_counts(normalize=True)

    def histoplot_accident_by_phase_of_flight(self):
        """Creates histplot that showes number of accidents by phase of flight"""

        phase_order = ['Standing', 'Taxi', 'Takeoff', 'Climb', 'Cruise', 'Descent', 'Approach', 'Go-around', 'Landing', 'Maneuvering', 'Other', 'Unknown']
        phase = pd.DataFrame({'Broad.phase.of.flight' : self.__data_set['Broad.phase.of.flight']})
        phase['Broad.phase.of.flight'] = pd.Categorical(phase['Broad.phase.of.flight'], categories=phase_order, ordered=True)
        sns.histplot(phase, x='Broad.phase.of.flight', discrete=True, color='lightsteelblue')
        plt.title("Accidents by phase of flight")
        plt.xlabel("Flight phase")
        plt.xticks(rotation=45)
        plt.show()

    def histplot_acciedent_year_hue_weather(self):
        """Creates histplot with accident numbers through years grouped by weather conditions"""

        injuries = pd.DataFrame({
            'Weather.Condition' : self.__data_set['Weather.Condition'],
            'Event.Year' : self.__data_set['Event.Date']
        })
        injuries['Event.Year'] = injuries['Event.Year'].dt.year
        injuries['Weather.Condition'] = injuries['Weather.Condition'].str.upper()
        sns.histplot(injuries, x='Event.Year', hue='Weather.Condition', stat='probability', palette=['seashell', 'slategrey', 'lightskyblue'], discrete=True)
        plt.xlim(1980, 2025)
        plt.title("Accidents through years")
        plt.show()

    def histplot_acciedent_month_hue_weather(self):
        """Creates histplot that shows number of accidents in each month grouped by weather conditions"""

        injuries = pd.DataFrame({
            'Weather.Condition' : self.__data_set['Weather.Condition'],
            'Event.Month' : self.__data_set['Event.Date']
        })
        injuries['Event.Month'] =  pd.Categorical(injuries['Event.Month'].dt.month_name(), categories=month_name[1:], ordered=True)
        injuries['Weather.Condition'] = injuries['Weather.Condition'].str.upper()
        sns.histplot(injuries, x='Event.Month', hue='Weather.Condition', stat='probability', palette=['seashell', 'slategrey', 'lightskyblue'], discrete=True)
        plt.title("Accidents in each month depending on weather")
        plt.show()

    def __create_coordinates_indexes(self):
        """Set tuple of latitude and longtitude indexes from 90°S to 90°N and from 180°W to 180°E"""

        latitude_indexes = []
        for i in range(90, 0, -1):
            latitude_indexes.append(f'{i}°S')
        latitude_indexes.append('0°')
        for i in range(1, 91, 1):
            latitude_indexes.append(f'{i}°N')  
        longtitude_indexes = []
        for i in range(180, 0, -1):
            longtitude_indexes.append(f'{i}°W')
        longtitude_indexes.append('0°')
        for i in range(1, 181, 1):
            longtitude_indexes.append(f'{i}°E') 

        self._lat_long_indexes = latitude_indexes, longtitude_indexes

    def __create_coordinates_heatmap(self, raw_coordinates):
        """Creates a heatmap from tuple of coordinates - a tuple of int coordinates that might include invalid values"""

        latitude_raw, longtitude_raw = raw_coordinates

        if len(self._lat_long_indexes[0]) == 0 or len(self._lat_long_indexes[1]) == 0:
            self.__create_coordinates_indexes()

        coordinates = [[0 for j in range(361)] for i in range(181)]
        for i in range(len(latitude_raw)):
            if latitude_raw[i] != self.__invalid and longtitude_raw[i] != self.__invalid:
                coordinates[latitude_raw[i]+90][longtitude_raw[i]+180] += 1

        # normalized_data = np.log1p(coordinates)
        normalized_data = np.power(coordinates, 0.3)

        location = pd.DataFrame(normalized_data, columns=self._lat_long_indexes[1], index=self._lat_long_indexes[0])
        
        location_heatmap = sns.heatmap(location, cmap='Blues')

        location_heatmap.set_xticks(np.arange(0, len(self._lat_long_indexes[1]), 20))
        location_heatmap.set_xticklabels(self._lat_long_indexes[1][::20])

        location_heatmap.set_yticks(np.arange(0, len(self._lat_long_indexes[0]), 10))
        location_heatmap.set_yticklabels(self._lat_long_indexes[0][::10])

        # Invert Y axis(0 at the botttom)
        plt.gca().invert_yaxis()
        plt.xticks(rotation=45)
        plt.title("Location of accidents")

        plt.show()

    def heatmap_locations(self):
        """Convert latitude and longtitude values and pass it to function creaating heatmap"""

        coordinates_data = pd.read_csv('coordinates_data.csv', low_memory=False)

        latitude_raw = []
        longtitude_raw = []

        for coordinate in coordinates_data['latitude']:
            str_coordinate = str(coordinate)
            int_coordinate = self.__invalid
            if str_coordinate.lower() != 'nan':
                if str_coordinate[-1] == 'S':
                    int_coordinate = -1 * int(str_coordinate[0:-5])
                else:
                    int_coordinate = int(str_coordinate[0:-5])

                if int_coordinate != self.__invalid:
                    while int_coordinate >= 90 or int_coordinate <= -90:
                        int_coordinate //= 10
                    latitude_raw.append(int_coordinate)
            else:
                latitude_raw.append(self.__invalid)

        for coordinate in coordinates_data['longitude']:
            str_coordinate = str(coordinate)
            int_coordinate = self.__invalid
            if str_coordinate.lower() != 'nan':
                if str_coordinate[-1] == 'W':
                    int_coordinate = -1 * int(str_coordinate[0:-5])
                else:
                    int_coordinate = int(str_coordinate[0:-5])

                if int_coordinate != self.__invalid:
                    while int_coordinate >= 180 or int_coordinate <= -180:
                        int_coordinate //= 10
                    longtitude_raw.append(int_coordinate)
            else:
                longtitude_raw.append(self.__invalid)

        self.__create_coordinates_heatmap((latitude_raw, longtitude_raw))

    def heatmap_diff_data(self):
        """Convert latitude and longtitude values from another source of data and pass it to function creaating heatmap"""

        latitude_raw = []
        longtitude_raw = []

        for coordinate in self.__data_set['Latitude']:
            str_coordinate = str(coordinate)
            int_coordinate = self.__invalid
            if str_coordinate.lower() != 'nan':
                if '.' in str_coordinate:
                    int_coordinate = int(float(str_coordinate))
                elif str_coordinate[-1] == 'N' or str_coordinate[-1] == 'S':
                    if len(str_coordinate[0:-5]) > 0:
                        if str_coordinate[-1] == 'S':
                            int_coordinate = -1 * int(str_coordinate[0:-5])
                        else:
                            int_coordinate = int(str_coordinate[0:-5])
                else:
                    int_coordinate = int(str_coordinate)

            if int_coordinate <= 90 and int_coordinate >= -90:
                latitude_raw.append(int_coordinate)
            else:
                latitude_raw.append(self.__invalid)

        for coordinate in self.__data_set['Longitude']:
            str_coordinate = str(coordinate)
            int_coordinate = self.__invalid
            if str_coordinate.lower() != 'nan':
                if '.' in str_coordinate:
                    int_coordinate = int(float(str_coordinate))
                elif str_coordinate[-1] == 'W' or str_coordinate[-1] == 'E':
                    if len(str_coordinate[0:-5]) > 0:
                        if str_coordinate[-1] == 'W':
                            int_coordinate = -1 * int(str_coordinate[0:-5])
                        else:
                            int_coordinate = int(str_coordinate[0:-5])
                else:
                    int_coordinate = int(str_coordinate)

            if int_coordinate <= 180 and int_coordinate >= -180:
                longtitude_raw.append(int_coordinate)
            else:
                longtitude_raw.append(self.__invalid)

            self.__create_coordinates_heatmap((latitude_raw, longtitude_raw))

    def violinplot_fatalities_percentage(self):
        """Creates a violinplot that shows percentage of fatalities in fatal accidents with more that 10 souls on board"""

        dead_souls = pd.DataFrame({
            'Total.Souls' : self.__data_set['Total.Fatal.Injuries'] + self.__data_set['Total.Serious.Injuries'] + self.__data_set['Total.Minor.Injuries'] + self.__data_set['Total.Uninjured'],
            'Total.Fatal.Injuries' : self.__data_set['Total.Fatal.Injuries']
        })

        fatalities = []
        for dead, souls in zip(dead_souls['Total.Fatal.Injuries'], dead_souls['Total.Souls']):
            if souls > 10 and not isnan(souls) and not isnan(dead):
                pct = round(dead/souls *100, 2)
                if pct != 0:
                    fatalities.append(pct)

        plot = sns.violinplot(fatalities, cut=0, color='maroon', inner='sticks')   
        plt.title("Fatalities in aviation crashes")  
        plt.ylabel("Percentage")
        plt.show()

    def boxplot_injuries_percentage(self):
        """Creates a boxplot that shows percentage of injuries in accidents with injured people and more that 10 souls on board,
        grouped by type of investigation"""

        injured_souls_invest_type = pd.DataFrame({
            'Total.Souls' : self.__data_set['Total.Fatal.Injuries'] + self.__data_set['Total.Serious.Injuries'] + self.__data_set['Total.Minor.Injuries'] + self.__data_set['Total.Uninjured'],
            'Total.Injuries' : self.__data_set['Total.Fatal.Injuries'] + self.__data_set['Total.Serious.Injuries'] + self.__data_set['Total.Minor.Injuries'],
            'Investigation.Type' : self.__data_set['Investigation.Type'],
        })

        injuries_by_investugation_type = {
            'Investigation.Type' : [],
            'Injured.Percentage' : []
        }
        for injured, souls, inv_type in zip(injured_souls_invest_type['Total.Injuries'], injured_souls_invest_type['Total.Souls'], injured_souls_invest_type['Investigation.Type']):
            if souls > 10 and not isnan(souls) and not isnan(injured):
                pct = round(injured/souls *100, 2)
                if pct != 0:
                    injuries_by_investugation_type['Investigation.Type'].append(inv_type)
                    injuries_by_investugation_type['Injured.Percentage'].append(pct)

        injuries = pd.DataFrame.from_dict(injuries_by_investugation_type)
        plot = sns.boxplot(injuries, x='Injured.Percentage', y='Investigation.Type', width=.5, color='firebrick', whis=1)       
        plt.grid(True, axis='x')
        plt.title("Injuries in aviation crashes")
        plt.show()

    def regplot_accidents_through_years(self):
        """Creates a regplot that shows number of accidents in each year from 1982"""

        accidents = []
        min_year = 1982
        for i in range(min_year, 2023):
            accidents.append([i, 0])
        
        for date in self.__data_set['Event.Date']:
            if date.year >= min_year:
                accidents[date.year-min_year][1] += 1

        accidents_by_year = pd.DataFrame(accidents, columns=['Event.Year', 'Accidents.Number'])
        sns.regplot(accidents_by_year, x='Event.Year', y='Accidents.Number', order=3, scatter_kws={'color' : 'darkblue'}, line_kws={'color' : 'royalblue'})
        plt.title("Number of accidents through years")
        plt.show()

    def errorbar_time_to_report(self):
        """Returns a striplot with pointplot and errorbar that shows how many years was needed to publish a report"""

        number_of_days = []
        for event, publication in zip(self.__data_set['Event.Date'], self.__data_set['Publication.Date']):
            if pd.notnull(event) and pd.notnull(publication):
                number_of_days.append(publication - event)

        time_from_accident_to_report = pd.DataFrame(number_of_days, columns=['Number.of.Days'])
        time_from_accident_to_report['Number.of.Days'] = time_from_accident_to_report['Number.of.Days'].dt.days

        sns.stripplot(data=time_from_accident_to_report, x='Number.of.Days', alpha=.05, jitter=.2, color='lightslategrey')
        sns.pointplot(data=time_from_accident_to_report, x='Number.of.Days', errorbar='pi', capsize=0.5, color='navy', err_kws={'linewidth': 3})
        x_ticks = plt.xticks()[0]
        year_labels = [f"{int(day // 365)}" for day in x_ticks]
        plt.xticks(ticks=x_ticks, labels=year_labels)
        plt.title("Number of years between event and publication of the report")
        plt.xlabel("Number.of.Years")
        plt.show()

    def pca(self):
        """Returns a plot that was made using PCA dimension reduction"""

        numerical_aviation_data = pd.DataFrame({
            'Number.of.Engines' : self.__data_set['Number.of.Engines'],
            'Total.Fatal.Injuries' : self.__data_set['Total.Fatal.Injuries'].apply(lambda x: x * 1000),
            'Total.Serious.Injuries' : self.__data_set['Total.Serious.Injuries'].apply(lambda x: x * 100),
            'Total.Minor.Injuries' : self.__data_set['Total.Minor.Injuries'].apply(lambda x: x * 10),
            'Total.Uninjured' : self.__data_set['Total.Uninjured'],
            'Injury.Severity' : self.__data_set['Injury.Severity'],
            'Aircraft.damage' : self.__data_set['Aircraft.damage']
        })

        numerical_aviation_data['Injury.Severity'] = numerical_aviation_data['Injury.Severity'].str.replace(r'\(\d+\)', '', regex=True)

        injury_severity_codes = {
            'Unavailable' : -1,
            'Incident' : 0,
            'Non-Fatal' : 1,
            'Minor' : 2,
            'Serious' : 3,
            'Fatal' : 4
        }

        aircraft_damgae_codes = {
            'Unknown' : 0,
            'Minor' : 1,
            'Substantial' : 2,
            'Destroyed' : 3
        }

        pd.set_option('future.no_silent_downcasting', True) 
        numerical_aviation_data['Injury.Severity'] = numerical_aviation_data['Injury.Severity'].replace(injury_severity_codes)
        numerical_aviation_data['Aircraft.damage'] = numerical_aviation_data['Aircraft.damage'].replace(aircraft_damgae_codes)

        numerical_aviation_data_pca = numerical_aviation_data[numerical_aviation_data.columns].dropna()

        X = numerical_aviation_data_pca.iloc[:, 0:6].values
        y = numerical_aviation_data_pca.iloc[:, 6].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

        sc = StandardScaler()

        X_train = sc.fit_transform(X_train)

        X_train = pd.DataFrame(X_train, columns=['Number.of.Engines', 'Total.Fatal.Injuries', 'Total.Serious.Injuries',
             'Total.Minor.Injuries', 'Total.Uninjured',  'Injury.Severity'])

        pca = PCA(n_components=2)

        X_pca = pca.fit_transform(X_train)

        fig = plt.figure()

        colormap = plt.cm.get_cmap('YlGnBu')

        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap=colormap)

        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')

        plt.colorbar(scatter, label="Aircraft damage")

        plt.show()