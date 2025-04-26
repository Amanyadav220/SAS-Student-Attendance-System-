from flask_mysqldb import MySQL

mysql = MySQL()

def get_registered_students_from_db():
    cur = mysql.connection.cursor()
    cur.execute("SELECT name, student_id FROM students")  # Update table/column names if needed
    rows = cur.fetchall()
    students = [{'name': row[0], 'id': row[1]} for row in rows]
    cur.close()
    return students
