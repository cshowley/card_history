'''
features

26w moving average price
12w moving average price
200d moving average price (may be NaN unless we include data before 9/2025)
50d moving average price
last sale price
previous 1 week average price
previous 2 week average price
previous 3 week average price
previous 4 week average price
grade
grading company
time series-derived price to current day
round half grades down to floor grade
	include a binary feature for half grade or pristine/black cards
	1 if half/pristine (e.g. 8.5)
	0 if not half/pristine
	
(all values 1 grade below target)
26w moving average price
12w moving average price
200d moving average price (may be NaN unless we include data before 9/2025)
50d moving average price
macd value
last sale price
previous 1 week average price
previous 2 week average price
previous 3 week average price
previous 4 week average price
time series-derived price to current day
grading company (falls back to next-highest volume grader)

(all values 1 grade above target)
26w moving average price
12w moving average price
200d moving average price (may be NaN unless we include data before 9/2025)
50d moving average price
macd value
last sale price
previous 1 week average price
previous 2 week average price
previous 3 week average price
previous 4 week average price
time series-derived price to current day
grading company (falls back to next-highest volume grader)

**market price action, defined as top N cards and/or an index of popular cards
previous 1 week price % change from previous week
volume (number of sales)
previous 2 week price % change from previous week
volume (number of sales)
previous 3 week price % change from previous week
volume (number of sales)
previous 4 week price % change from previous week
volume (number of sales)


'''