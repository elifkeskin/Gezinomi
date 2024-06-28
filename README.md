# Gezinomi
 Rule-Based Classification 
Business Problem:

Gezinomi uses some features of its sales to make level-based sales.

Goal: Segmenting customers according to new sales definitions.

Gezinomi wants to guess that How much money can prospective customers bring to the company on average?

For example: Those who want to go to an All Inclusive hotel from Antalya during a busy period
determining how much money a customer can earn on average is wanted.


Dataset Story:

The data set consists of records created in each sales transaction. This means the table is not deduplicated. In other words. The customer may have made more than one purchase.


Data Fields:

SaleId: Sale Id
SaleDate : Sale Date
Price: Price paid for sale.
ConceptName:Hotel concept information.
SaleCityName:Information about the city where the hotel is located.
CheckInDate: Customer's hotel check-in date.
CInDay:The day the customer checks in to the hotel.
SaleCheckInDayDiff: Day difference between check-in at the hotel and check-in date in booking.
Season:Season information on hotel check-in date.
