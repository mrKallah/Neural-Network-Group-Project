import pandas as pd
from datetime import date


def read_data(file_name):
    data_frame = pd.read_csv(file_name, low_memory=False)

    return data_frame


def split_state_holiday_column(data_frame):
    bank_holiday = list()
    easter = list()
    christmas = list()
    no_holiday = list()

    for data in data_frame['StateHoliday']:
        if data is "a":
            bank_holiday.append(1)
            easter.append(0)
            christmas.append(0)
            no_holiday.append(0)
        elif data is "b":
            bank_holiday.append(0)
            easter.append(1)
            christmas.append(0)
            no_holiday.append(0)
        elif data is "c":
            bank_holiday.append(0)
            easter.append(0)
            christmas.append(1)
            no_holiday.append(0)
        elif data is "0":
            bank_holiday.append(0)
            easter.append(0)
            christmas.append(0)
            no_holiday.append(1)
    return bank_holiday, easter, christmas, no_holiday


def convert_date_to_int(string):
    split_date = str(string).split('-')

    first_date = date(1900, 1, 1)
    date_to_convert = date(int(split_date[0]), int(split_date[1]), int(split_date[2]))

    delta = date_to_convert - first_date

    return delta.days


def handle_date_format(date_column):
    new_date = list()

    for line_data in date_column:
        new_date.append(convert_date_to_int(str(line_data)))

    return new_date


def create_empty_column(row_count):
    empty_column = list()

    for rows in range(row_count):
        empty_column.append(0)

    return empty_column


def handle_data(data_frame, isX):
    bank_holiday, easter, christmas, no_holiday = split_state_holiday_column(data_frame)

    new_date = handle_date_format(data_frame["Date"])

    if isX == 0:
        sales = data_frame["Sales"]
        customers = data_frame["Customers"]
    elif isX == 1:
        sales = create_empty_column(row_count=data_frame.shape[0])
        customers = create_empty_column(row_count=data_frame.shape[0])

    raw = {'Store': data_frame["Store"], 'DayOfWeek': data_frame["DayOfWeek"], 'Date': new_date,
           'Sales': sales, 'Customers': customers, 'Open': data_frame["Open"],
           'Promo': data_frame["Promo"], 'BankHoliday': bank_holiday, 'Easter': easter, 'Christmas': christmas,
           'NoHoliday': no_holiday, 'SchoolHoliday': data_frame["SchoolHoliday"]}

    return_data_frame = pd.DataFrame(raw, columns=['Store', 'DayOfWeek', 'Date', 'Sales', 'Customers', 'Open', 'Promo',
                                                   'BankHoliday', 'Easter', 'Christmas', 'NoHoliday', 'SchoolHoliday'])

    return return_data_frame


def save_data_to_csv(data_frame, file_name):
    data_frame.to_csv(file_name)


def main():
    train_data = read_data("train.csv")
    test_data = read_data("test.csv")

    y_train_data = handle_data(train_data, 0)
    X_train_data = handle_data(train_data, 1)
    X_test_data = handle_data(test_data, 1)

    save_data_to_csv(y_train_data, "y_train.csv")
    save_data_to_csv(X_train_data, "X_train.csv")
    save_data_to_csv(X_test_data, "X_test.csv")


main()