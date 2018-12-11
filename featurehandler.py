import pandas as pd
import numpy
from datetime import date


def read_data(file_name):
    """
    Reads data from a .csv file and loads it into a pandas data frame
    @author Ioana-Adina Panoschi
    :param file_name: the name of the file
    :return: the newly created data frame
    """
    data_frame = pd.read_csv(file_name, low_memory=False)

    return data_frame


def split_column(df, column_name, values, number_of_new_columns):
    """
    Given a data frame, a column name and a list of the unique values in said column, the function splits the column
    in as many one-hot columns as the number of unique variables
    @autohor: Ioana-Adina Panoschi
    :param df: the data frame
    :param column_name: the name of the column that will be split
    :param values: the list of unique values of the column
    :param number_of_new_columns: the number of unique variables
    :return: lists containing the replacing columns
    """
    column_a = list()
    column_b = list()
    column_c = list()
    column_d = list()

    for data in df[column_name]:
        if data == values[0]:
            column_a.append(1)
            column_b.append(0)
            column_c.append(0)
            column_d.append(0)
        elif data == values[1]:
            column_a.append(0)
            column_b.append(1)
            column_c.append(0)
            column_d.append(0)
        elif data == values[2]:
            column_a.append(0)
            column_b.append(0)
            column_c.append(1)
            column_d.append(0)
        elif data == values[3]:
            column_a.append(0)
            column_b.append(0)
            column_c.append(0)
            column_d.append(1)

    if number_of_new_columns == 3:
        return column_a, column_b, column_c

    return column_a, column_b, column_c, column_d


def convert_date_to_int(date_string):
    """
    Algorithm that thansofrms a date in the form of string into an integer
    @author: Ioana-Adina Panoschi
    :param date_string: date in the form of string
    :return: the transformed date, representing the number of days since a predefined start date (in this case 01/01/1900)
    """
    split_date = str(date_string).split('-')

    first_date = date(1900, 1, 1)
    date_to_convert = date(int(split_date[0]), int(split_date[1]), int(split_date[2]))

    delta = date_to_convert - first_date

    return delta.days


def handle_date_format(date_column):
    """
    For each date value found in a given date column, send the string form of the date to the function convert_date_to_int()
    @author: Ioana-Adina Panoschi
    :param date_column: the column containing date values
    :return: returns a list of the converted date values
    """
    new_date = list()

    for line_data in date_column:
        new_date.append(convert_date_to_int(str(line_data)))

    return new_date


def create_empty_column(row_count):
    """
    Creates an empty list (to be column) containing only values of 0; used for creating empty columns for test files
    @author: Ioana-Adina Panoschi
    :param row_count: the number of rows that the column should have
    :return: the list containing only values of 0
    """
    empty_column = list()

    for rows in range(row_count):
        empty_column.append(0)

    return empty_column


def handle_data(data_frame, isX):
    """
    Handles data from train.csv and test.csv files;
    First, it splits the column StateHolidays into 4 equivalen one-hot encoded columns
    Secondly, it converts date objects into integers
    Afterwards, if the received data frame is for creating X/test files, it creates empty columns
    Lastly, it creates a new data frame containing the modified columns
    @author: Ioana-Adina Panoschi
    :param data_frame: the data frame
    :param isX: constant value containing 1 or 0; 1 -> the data frame is used for creating an X file, 0 -> the data frame
                is used for creating an y file
    :return: the updated data frame
    """
    state_holidays = ["a", "b", "c", "0"]
    public_holiday, easter, christmas, no_holiday = split_column(data_frame, "StateHoliday", state_holidays, number_of_new_columns=4)

    new_date = handle_date_format(data_frame["Date"])

    if isX == 0:
        sales = data_frame["Sales"]
        customers = data_frame["Customers"]
    elif isX == 1:
        sales = create_empty_column(row_count=data_frame.shape[0])
        customers = create_empty_column(row_count=data_frame.shape[0])

    raw = {'Store': data_frame["Store"], 'DayOfWeek': data_frame["DayOfWeek"], 'Date': new_date,
           'Sales': sales, 'Customers': customers, 'Open': data_frame["Open"],
           'Promo': data_frame["Promo"], 'PublicHoliday': public_holiday, 'Easter': easter, 'Christmas': christmas,
           'NoHoliday': no_holiday, 'SchoolHoliday': data_frame["SchoolHoliday"]}

    return_data_frame = pd.DataFrame(raw, columns=['Store', 'DayOfWeek', 'Date', 'Sales', 'Customers', 'Open', 'Promo',
                                                   'PublicHoliday', 'Easter', 'Christmas', 'NoHoliday', 'SchoolHoliday'])

    return return_data_frame


def modify_store_data(store_data_frame):
    """
    Handles data from store.csv
    Splits the columns StoreType, Assortment and PromoInterval into one-hot encoded columns
    Creates new data frame containing the new columns
    @author: Ioana-Adina Panoschi
    :param store_data_frame: data frame containing the data from store.csv
    :return: modified data frame
    """
    store_types = ["a", "b", "c", "d"]
    type_a, type_b, type_c, type_d = split_column(store_data_frame, "StoreType", store_types, number_of_new_columns=4)

    store_assortments = ["a", "b", "c"]
    assortment_a, assortment_b, assortment_c = split_column(store_data_frame, "Assortment", store_assortments, number_of_new_columns=3)

    promo_intervals = ["Jan,Apr,Jul,Oct", "Feb,May,Aug,Nov", "Mar,Jun,Sept,Dec", "0"]
    jan_apr_jul_oct, feb_may_aug_nov, mar_jun_sept_dec, no = split_column(store_data_frame, "PromoInterval", promo_intervals, number_of_new_columns=4)

    raw_data = {'Store': store_data_frame["Store"],'StoreTypeA': type_a, 'StoreTypeB': type_b, 'StoreTypeC': type_c, 'StoreTypeD': type_d,
                'BasicAssortment': assortment_a, 'ExtraAssortment': assortment_b, 'ExtendedAssortment': assortment_c,
                'CompetitionDistance': store_data_frame["CompetitionDistance"], 'CompetitionOpenSinceMonth': store_data_frame["CompetitionOpenSinceMonth"],
                'CompetitionOpenSinceYear': store_data_frame["CompetitionOpenSinceYear"], 'Promo2': store_data_frame["Promo2"],
                'Promo2SinceWeek': store_data_frame["Promo2SinceWeek"], 'Promo2SinceYear': store_data_frame["Promo2SinceYear"],
                'PIJanAprJulOct': jan_apr_jul_oct, 'PIFebMayAugNov': feb_may_aug_nov, 'PIMarJunSeptDec': mar_jun_sept_dec}

    modified_data_frame = pd.DataFrame(raw_data, columns=['Store', 'StoreTypeA', 'StoreTypeB', 'StoreTypeC', 'StoreTypeD', 'BasicAssortment',
                                                          'ExtraAssortment', 'ExtendedAssortment', 'CompetitionDistance',
                                                          'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
                                                          'Promo2SinceYear', 'PIJanAprJulOct', 'PIFebMayAugNov', 'PIMarJunSeptDec'])

    return modified_data_frame


def append_store_data(t_data_frame, store_data_frame):
    """
    Function that appends the modified data collected from store.csv to a given data frame
    @author: Ioana-Adina Panoschi
    :param t_data_frame: data frame that the algorithm appends to
    :param store_data_frame: data frame containing data form store.csv
    :return: new data fram with the columns form store_data_frame appended to the t_data_frame
    """
    type_a = list(); type_b = list(); type_c = list(); type_d = list()
    assortment_a = list(); assortment_b = list(); assortment_c = list()
    competition_distance = list(); competition_open_month = list(); competition_open_year = list()
    promo2 = list(); promo2_since_week = list(); promo2_since_year = list()
    jan_apr_jul_oct = list(); feb_may_aug_nov = list(); mar_jun_sept_dec = list()

    for data in store_data_frame["Store"]:
        index_list = numpy.where(t_data_frame["Store"] == data)
        index_in_df = numpy.where(store_data_frame["Store"] == data)
        for index in list(index_list[0]):
            type_a.insert(index, store_data_frame["StoreTypeA"][index_in_df[0][0]])
            type_b.insert(index, store_data_frame["StoreTypeB"][index_in_df[0][0]])
            type_c.insert(index, store_data_frame["StoreTypeC"][index_in_df[0][0]])
            type_d.insert(index, store_data_frame["StoreTypeD"][index_in_df[0][0]])
            assortment_a.insert(index, store_data_frame["BasicAssortment"][index_in_df[0][0]])
            assortment_b.insert(index, store_data_frame["ExtraAssortment"][index_in_df[0][0]])
            assortment_c.insert(index, store_data_frame["ExtendedAssortment"][index_in_df[0][0]])
            competition_distance.insert(index, store_data_frame["CompetitionDistance"][index_in_df[0][0]])
            competition_open_month.insert(index, store_data_frame["CompetitionOpenSinceMonth"][index_in_df[0][0]])
            competition_open_year.insert(index, store_data_frame["CompetitionOpenSinceYear"][index_in_df[0][0]])
            promo2.insert(index, store_data_frame["Promo2"][index_in_df[0][0]])
            promo2_since_week.insert(index, store_data_frame["Promo2SinceWeek"][index_in_df[0][0]])
            promo2_since_year.insert(index, store_data_frame["Promo2SinceYear"][index_in_df[0][0]])
            jan_apr_jul_oct.insert(index, store_data_frame["PIJanAprJulOct"][index_in_df[0][0]])
            feb_may_aug_nov.insert(index, store_data_frame["PIFebMayAugNov"][index_in_df[0][0]])
            mar_jun_sept_dec.insert(index, store_data_frame["PIMarJunSeptDec"][index_in_df[0][0]])

    raw_data = {'StoreTypeA': type_a, 'StoreTypeB': type_b, 'StoreTypeC': type_c, 'StoreTypeD': type_d,
                'BasicAssortment': assortment_a, 'ExtraAssortment': assortment_b, 'ExtendedAssortment': assortment_c,
                'CompetitionDistance': competition_distance, 'CompetitionOpenSinceMonth': competition_open_month,
                'CompetitionOpenSinceYear': competition_open_year, 'Promo2': promo2, 'Promo2SinceWeek': promo2_since_week,
                'Promo2SinceYear': promo2_since_year, 'PIJanAprJulOct': jan_apr_jul_oct, 'PIFebMayAugNov': feb_may_aug_nov,
                'PIMarJunSeptDec': mar_jun_sept_dec}

    modified_data_frame = pd.DataFrame(raw_data, columns=['StoreTypeA', 'StoreTypeB', 'StoreTypeC', 'StoreTypeD',
                                                          'BasicAssortment', 'ExtraAssortment', 'ExtendedAssortment',
                                                          'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear',
                                                          'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear',
                                                          'PIJanAprJulOct', 'PIFebMayAugNov', 'PIMarJunSeptDec'])

    t_data_frame = t_data_frame.join(modified_data_frame)
    return t_data_frame


def save_data_to_csv(data_frame, file_name):
    """
    Function that saves the data from a given data frame to a file with a given file
    @author: Ioana-Adina Panoschi
    :param data_frame: data frame that will be saved to a file
    :param file_name: name of the file 
    :return: no return
    """
    data_frame.to_csv(file_name, index=False, index_label=False)


def main():
    train_data_frame = read_data("original/train.csv")
    test_data_frame = read_data("original/test.csv")
    store_data_frame = read_data("original/store.csv")

    train_data_frame = train_data_frame.replace(numpy.nan, "0")
    test_data_frame = test_data_frame(numpy.nan, "0")
    store_data_frame = store_data_frame.replace(numpy.nan, "0")

    modified_store_data_frame = modify_store_data(store_data_frame)

    y_train_data = handle_data(train_data_frame, 0)
    X_test_data = handle_data(test_data_frame, 1)
    X_train_data = handle_data(train_data_frame, 1)

    appended_y_train = append_store_data(y_train_data, modified_store_data_frame)
    appended_x_train = append_store_data(X_train_data, modified_store_data_frame)
    appended_x_test = append_store_data(X_test_data, modified_store_data_frame)

    save_data_to_csv(appended_y_train, "data_files/y_train.csv")
    save_data_to_csv(appended_x_train, "data_files/X_train.csv")
    save_data_to_csv(appended_x_test, "data_files/X_test.csv")


main()
