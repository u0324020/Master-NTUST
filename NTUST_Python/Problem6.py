#problem six coding by Jane

import MySQLdb
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column
from sqlalchemy import String
from sqlalchemy import Integer

BaseModel = declarative_base()
engine = create_engine("mysql://root:0324020@localhost/NTUST_StudentSystem", pool_recycle=5)

class Students(BaseModel):
    __tablename__ = "Students"
    sn = Column(Integer, primary_key=True)
    ID = Column("ID", String(50))
    grades = Column("grades", String(50))

BaseModel.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session2 = Session()

def Show():
    conn = engine.connect()
    rows = session2.execute("SELECT ID, grades FROM NTUST_StudentSystem.Students;").fetchall()
    print ("Info:")
    for row in rows:
    	print row

    conn.close()
    return 0

def Revise_ONE(action, action_ID, action_grades):
	action_ID = str(action_ID)
	action_grades = str(action_grades)
	conn = engine.connect()
	ans = session2.execute("SELECT COUNT(1) FROM NTUST_StudentSystem.Students WHERE ID ='"+action_ID+"';")

	if action == 1: #INSERT
		#ans = session2.execute("SELECT COUNT(1) FROM NTUST_StudentSystem.Students WHERE ID ='"+action_ID+"';")
		if ans.fetchone()[0]:
			print "* Error : This ID already exists *"
			onn.close()
		else:
			session2.execute("INSERT into NTUST_StudentSystem.Students(ID, grades) values ('"+action_ID+"','"+action_grades+"');")
			session2.commit()
			conn.close()

	if action == 3: #UPDATE
		#ans = session2.execute("SELECT COUNT(1) FROM NTUST_StudentSystem.Students WHERE ID ='"+action_ID+"';")
		if ans.fetchone()[0]:
			session2.execute("UPDATE NTUST_StudentSystem.Students SET grades ='"+action_grades+"' WHERE ID = '"+action_ID+"' ;")
			session2.commit()
			conn.close()
		else:
			print "* Error : This ID not exists *"

def Revise_TWO(action, action_ID):
	action_ID = str(action_ID)
	conn = engine.connect()
	ans = session2.execute("SELECT COUNT(1) FROM NTUST_StudentSystem.Students WHERE ID ='"+action_ID+"';")
	
	if action == 2: #DELETE
		if ans.fetchone()[0]:
			session2.execute("DELETE from NTUST_StudentSystem.Students WHERE ID ='"+action_ID+"';")
			session2.commit()
			conn.close()
		else:
			print "* Error : This ID not exists *"
		Show()

	if action == 4: #SEARCH
		if ans.fetchone()[0]:
			rows = session2.execute("SELECT grades from NTUST_StudentSystem.Students WHERE ID ='"+action_ID+"';").fetchall()
			print ("grades is %s"%(rows))
			conn.close()
		else:
			print "* Error : This ID not exists *"


if __name__ == '__main__':
	Show()
	act = input('1)insert 2)delete 3)update 4)search :')
	act_ID = input('Enter the ID:')
	if act%2 == 0:
		Revise_TWO(act, act_ID)
	else:
		act_grades = input('Enter the grades:')
		Revise_ONE(act, act_ID, act_grades)
		Show()