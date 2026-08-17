from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

stmt1="This si compact computer institute"
stmt2="This is ML, we may g for Deep Learning"  
stmt3=" List of students : Bansari,Mudra,Prem,Jay,Jainam,Rudra,Dheyey,Sandeep,Abhinav"

mydf=pd.DataFrame({'First_Stmt' : [stmt1],'Second_Stmt' : [stmt2], 'Third_Stmt' : [stmt3]})
print(mydf.iloc[0])
tdidf_vectorizer= TfidfVectorizer()
doc_vec= tdidf_vectorizer.fit_transform(mydf.iloc[0])
print(doc_vec)
print(doc_vec.toarray())
print(doc_vec.toarray().transpose())
