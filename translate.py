import csv

from google_trans_new import google_translator
def translate(row):
	translator = google_translator()  
	return translator.translate(row,lang_src='en', lang_tgt='it')  

counter=3450
with open('t.csv','r') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	with open('train_italian.csv', mode='a',newline="") as w:
		csv_writer=csv.writer(w)
		for line in csv_reader:
			l=[]
			counter+=1
			print(counter)
			for w in line:
				if w.isnumeric():
					l.append(w)
				else:
					l.append(translate(w))
			csv_writer.writerow(l)				
				

