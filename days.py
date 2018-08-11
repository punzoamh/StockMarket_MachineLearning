import time
import calendar
import datetime
from dateutil import rrule
from datetime import date, timedelta
def main():
    text_file = open("start_dates.txt", "w")
    date_start_obj = date(2014, 3, 17)
    date_end_obj = date(2015, 3, 17)
    days = 100
    """
    Find_dates_cpy() will get the start date
    for a given numner of days and a given end date
    """
    #print find_dates_cpy(300, date_start_obj)
    """
    From the date that find_dates_cpy() returns to
    date_start_obj is 300 days
    """
    #print get_working_days(find_dates_cpy(300, date_start_obj),date_start_obj)
    while(days < 5000):
        text_file.write(str(days) + ' ' + str(find_dates_cpy(days,date_end_obj)) + '\n')
        print days, ' ', find_dates_cpy(days,date_end_obj)
        days += 100

    text_file.close()

    
"""
Gets number of days excluding weekends between two dates
"""
def get_working_days(date_start_obj, date_end_obj):
    weekdays = rrule.rrule(rrule.DAILY, byweekday=range(0, 5), dtstart=date_start_obj, until=date_end_obj)
    weekdays = len(list(weekdays))
    if int(time.strftime('%H')) >= 18:
        weekdays -= 1
    return weekdays

"""
Finds the start date given a number of days and the end date
"""
def find_dates_cpy(days, end_date):
    end_date = end_date
    days = days
    month = end_date.month
    year = end_date.year
    the_day = end_date.day
    start_date = date(year,month,the_day)

    while(days > get_working_days(start_date,end_date)):
        start_date = start_date - datetime.timedelta(days=1)
        #print get_working_days(start_date,end_date)
        #print start_date, end_date
    #print start_date
    return start_date

"""
Gets the last day of the given month of a given year
The input for this function will be calendar.monthrange(year,month)
but we don't really need this anymore
"""
def get_day_month(dates):
    dates = str(dates)


    days = dates[4] + dates[5]
    return days



main()
