from yahoo_finance import Share
from pprint import pprint

#Stock Holders#
apple = Share('AAPL')
google = Share('GOOG')
cisco = Share('CSCO')
yahoo = Share('YHOO')
microsoft = Share('MSFT')

print("Started Importing . . . ")

print("starting apple")
#Apple Import
apple.get_price()
#historical = str(apple.get_historical('2014-06-01','2015-01-01'))
historical = str(google.get_historical('2015-01-01','2015-06-01'))
file = open("apple.txt","wb")
file.write(historical)
file.close()

"""
print("starting google")
#Google Import
google.get_price()
#historical = str(google.get_historical('2014-06-01','2015-01-01'))
historical = str(google.get_historical('2015-01-01','2015-06-01'))
file = open("google.txt","wb")
file.write(historical)
file.close()

print("starting yahoo")
#Yahoo Import
yahoo.get_price()
historical = str(yahoo.get_historical('2014-06-01','2015-01-01'))
file = open("yahoo.txt","wb")
file.write(historical)
file.close()

print("starting cisco")
#Cisco Import
cisco.get_price()
historical = str(cisco.get_historical('2014-06-01','2015-01-01'))
file = open("cisco.txt","wb")
file.write(historical)
file.close()

print("starting microsoft")
#Microsoft Import
microsoft.get_price()
historical = str(microsoft.get_historical('2014-06-01','2015-01-01'))
file = open("microsoft.txt","wb")
file.write(historical)
file.close()
"""
