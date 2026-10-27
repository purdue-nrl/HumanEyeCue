import mysql.connector  # For MySQL database connection and operations
from mysql.connector import errorcode as sql_error_code  # To handle specific MySQL error codes
from mysql.connector import Error as sql_error  # To catch MySQL errors
import re  # For regular expression operations
import hashlib  # For hashing operations, specifically SHA-256 hashing
import datetime  # For handling date and time objects
import json
import heapq
import random
import logging
import os
import pandas as pd

'''
Define a class named SQLHandler to handle SQL operations
'''
class SQLHandler:
    # Constructor with a parameter `config` which is expected to be a dictionary containing database configuration
    def __init__(self, config, logger=None, dev=False, use_root=False) -> None:
        # Call to the super class constructor is unnecessary here since it's inheriting from `object` by default in Python 3
        super().__init__()

        self.database = config["export_database_dev"] if dev else config["export_database_prod"]

        # Store SQL configuration details extracted from `config` parameter
        self.sql_config = {
            "user": config["sql_user_root"] if use_root else config["sql_user"],
            "password": os.environ["HUMANEYECUE_DB_ROOT_PASSWORD" if use_root else "HUMANEYECUE_DB_PASSWORD"],
            "host": config["sql_url"],
            "database": self.database,
            "port": config["sql_port"],
            "raise_on_warnings": True,  # Configuration to raise exceptions on warnings
        }

        os.makedirs("logs", exist_ok=True)
        self.video_gap = config["video_gap"]
        if logger:
            self.logger = logger
        else:
            logger = logging.getLogger(f'SQL Logger')
            logger.setLevel(logging.INFO)
            handler = logging.FileHandler(os.path.join('./logs', f'sql_default.log'))
            formatter = logging.Formatter(
                fmt=f'%(asctime)s %(levelname)-8s %(message)s ',
                datefmt='%Y-%m-%d %H:%M:%S')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            self.logger = logger

        logger.info(f"Using Database {self.database}")

    # Method to execute a SQL query
    def execute_sql_query(self, query, commit=True, conn=None):
        """Executes a SQL query string on the MySQL database.

        Args:
            query (string): SQL query string to be executed.
            commit (bool): Whether to commit the transaction. Defaults to True.
            conn: Existing database connection to reuse. If None, a new connection is created.

        Returns:
            If commit is True, returns list of rows containing the query result.
            If commit is False, returns a tuple of query result and connection object.
        """
        res = None  # Initialize result variable
        try:
            if conn is None:  # Check if an existing connection is passed, if not, open a new one
                conn = mysql.connector.connect(**self.sql_config)  # Unpack sql_config to connect
        except mysql.connector.Error as err:  # Catch MySQL-specific errors
            # Error handling based on error codes
            if err.errno == sql_error_code.ER_ACCESS_DENIED_ERROR:
                self.logger.error("Something is wrong with your user name or password")
            elif err.errno == sql_error_code.ER_BAD_DB_ERROR:
                self.logger.error("Database does not exist")
            else:
                self.logger.error(err)
        else:
            try:
                cursor = conn.cursor()  # Create a cursor object to execute the query
                cursor.execute(query)  # Execute the SQL query
                res = cursor.fetchall()  # Fetch all rows of the query result
                if commit:  # If commit flag is true, commit the transaction
                    conn.commit()
            except sql_error as sql_err:  # Catch SQL errors during query execution
                print(sql_err)
            finally:
                cursor.close()  # Close the cursor
                if commit:  # If commit flag is true, close the connection
                    conn.close()

        # Return results based on the commit flag
        if not commit:
            return res, conn  # Return query result and connection object if commit is False
        return res  # Return only the query result if commit is True

    # Method to format a datetime object into a SQL-compatible datetime string
    def get_sql_datetime(self, timestamp: datetime):
        """Formats a datetime object to a string in SQL datetime format.

        Args:
            timestamp (datetime): The datetime object to format.

        Returns:
            str: A string representation of `timestamp` in SQL datetime format.
        """
        return timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")

    # Method to validate an email address format and its registration status
    def validate_email(self, email):
        """Validates an email address.

        Args:
            email (str): The email address to validate.

        Returns:
            bool: True if the email is valid and registered, False otherwise.
        """
        # Call `is_valid_email` and `is_email_registered` methods and return True if both checks pass
        if self.is_valid_email(email) and self.is_email_registered(email):
                return True

        return False

    # Method to check if an email address matches a simple regular expression pattern
    def is_valid_email(self, email):
        """Checks if the given email address is valid based on a regex pattern.
           Simple regex for checking the basic structure of an email address
           This pattern checks for a sequence of characters that does not include '@', followed by '@',
           then another sequence of characters that does not include '@', followed by '.',
           and finally, a sequence of characters that does not include '@'.
           Note: This is a very basic validation and might not cover all valid email addresses.
        Args:
            email (str): The email address to check.

        Returns:
            bool: True if the email matches the pattern
        """
        if re.match(r"[^@]+@[^@]+\.[^@]+", email):
            return True
        return False

    # Method to insert session data into the database
    def update_session_info(self, user_id):
        """Updates the session count for user
        """

        update_query = f"UPDATE EYE_TRACK_DB.REGISTERED SET sessions = sessions + 1 WHERE user_id = '{user_id}'"
        # Execute the query to update the number of responses count
        res = self.execute_sql_query(query=update_query, commit=True)

    # Method to insert session data into the database
    def send_info(
            self,
            user_id,
            video_no,
            question_no,
            q_start_time: datetime.datetime,
            q_stop_time: datetime.datetime,
            v_start_time: datetime.datetime,
            v_stop_time: datetime.datetime,
            a_start_time: datetime.datetime,
            a_stop_time: datetime.datetime,
            q_start_tick: int,
            q_stop_tick: int,
            v_start_tick: int,
            v_stop_tick: int,
            a_start_tick: int,
            a_stop_tick: int,
            answer,
            qtype):
        """Inserts session data into the database.
        """
        # Get human-readible user ID
        hr_uid = self.get_hruid_for_userid(user_id=user_id)
        # Convert startTime and stopTime from datetime objects to strings in SQL datetime format
        q_start_time_string = self.get_sql_datetime(q_start_time)
        q_stop_time_string = self.get_sql_datetime(q_stop_time)
        v_start_time_string = self.get_sql_datetime(v_start_time)
        v_stop_time_string = self.get_sql_datetime(v_stop_time)
        a_start_time_string = self.get_sql_datetime(a_start_time)
        a_stop_time_string = self.get_sql_datetime(a_stop_time)
        # Construct the SQL INSERT query
        query = 'INSERT INTO EYE_TRACK_DB.QVA_SESSION_DATA (user_id, hr_uid, video_no, question_no, q_start, q_stop, v_start, v_stop, a_start, a_stop, answer, qtype, q_start_tick, q_stop_tick, v_start_tick, v_stop_tick, a_start_tick, a_stop_tick, exp_q, exp_v, exp_a) VALUES' \
            + f"('{user_id}', {hr_uid}, {video_no}, {question_no}, '{q_start_time_string}', '{q_stop_time_string}', '{v_start_time_string}', '{v_stop_time_string}', '{a_start_time_string}', '{a_stop_time_string}', '{json.dumps(answer)}', '{qtype}', {q_start_tick}, {q_stop_tick}, {v_start_tick}, {v_stop_tick}, {a_start_tick}, {a_stop_tick}, FALSE, FALSE, FALSE);"
        # Execute the query
        res = self.execute_sql_query(query=query, commit=True)
        # No need to return anything; the function performs a database operation

        update_query = f"UPDATE VQDB.VQ_DATA SET num_responses = num_responses + 1 WHERE video_no = {video_no} AND question_no = {question_no}"
        # Execute the query to update the number of responses count
        res = self.execute_sql_query(query=update_query, commit=True)

    def get_video_questions_unanswered(self, user_id):
        query = "SELECT v.video_no, v.question_no, v.num_responses FROM VQDB.VQ_DATA v LEFT JOIN EYE_TRACK_DB.QVA_SESSION_DATA u " \
            + f"ON v.video_no = u.video_no AND v.question_no = u.question_no AND u.user_id = '{user_id}'" \
            + "WHERE u.video_no IS NULL AND u.question_no IS NULL;"
        # Execute the query without committing (since it's a SELECT query)
        unanswered, _ = self.execute_sql_query(query=query, commit=False)
        return unanswered

    def get_questions_answered(self):
        query = "SELECT v.video_no, v.question_no, v.num_responses FROM VQDB.VQ_DATA v where v.num_responses > 0"
        # Execute the query without committing (since it's a SELECT query)
        unanswered, _ = self.execute_sql_query(query=query, commit=False)
        return unanswered

    def schedule(self, questions, max_length=120):
        k = self.video_gap
        next_schedule_time = {}
        qs = {}
        for video_no, question_no, count in questions:
            next_schedule_time[video_no] = 0
            if video_no in qs:
                qs[video_no] += [(count, question_no)]
            else:
                qs[video_no] = [(count, question_no)]

        # Build a priority queue for each video
        for video_id, v in qs.items():
            random.shuffle(v)
            v_new = []
            for idx, (count, q_no) in enumerate(v):
                v_new.append((count, idx, q_no))
            heapq.heapify(v_new)
            qs[video_id] = v_new

        question_schedule = []
        scheduled = 0
        while scheduled < max_length:
            # Find a video id that can be scheduled
            # Pick a video which has the least number of responses
            min_responses = float('Inf') # Large value
            video_id_2_sch = -1
            for video_id, schedule_time in next_schedule_time.items():
                if schedule_time <= len(question_schedule) and video_id in qs:
                    responses, _, _ = qs[video_id][0]
                    if responses < min_responses:
                        min_responses = responses
                        video_id_2_sch = video_id

            if video_id_2_sch > -1:
                _, _, question = heapq.heappop(qs[video_id_2_sch])
                if len(qs[video_id_2_sch]) == 0:
                    qs.pop(video_id_2_sch)
                question_schedule.append((video_id_2_sch, question))
                scheduled += 1
                next_schedule_time[video_id_2_sch] = len(question_schedule) + k
            else:
                raise ValueError("Unable to schedule")

        return question_schedule

    def get_qva_data_for_question(self, video_id, question_id):
        query = f"SELECT * FROM {self.database}.QVA_SESSION_DATA where video_no = {video_id} AND question_no = {question_id}"
        result, conn = self.execute_sql_query(query=query, commit=False)
        conn.close()

        # Define the column names as per the given table schema
        column_names = [
            "qva_session_id", "user_id", "hr_uid", "video_no", "question_no", "attempt_no",
            "q_start", "q_stop", "v_start", "v_stop", "a_start", "a_stop",
            "answer", "qtype", "q_start_tick", "q_stop_tick", "v_start_tick",
            "v_stop_tick", "a_start_tick", "a_stop_tick", "exp_q", "exp_v", "exp_a"
        ]

        # Create DataFrame
        result_df = pd.DataFrame(result, columns=column_names)
        return result_df

    # Method to check if an email is already registered in the database
    def is_email_registered(self, email):
        """Checks if the given email is registered in the database.

        Args:
            email (str): The email address to check.

        Returns:
            tuple: A tuple containing a boolean indicating if the email is registered and the email's SHA-256 hash.
        """
        # Hash the email using SHA-256 and get the hexadecimal representation of the hash
        email_sha = hashlib.sha256(email.encode()).hexdigest()
        # Construct the SQL query to search for the hashed email in the EYE_TRACK_DB.REGISTERED table
        query = f"SELECT * FROM {self.database}.REGISTERED WHERE user_id = '{email_sha}'"
        # Execute the query without committing (since it's a SELECT query)
        res, _ = self.execute_sql_query(query=query, commit=False)
        # Check if the query returned any results
        valid = True if res else False
        # Return a tuple of the validation result and the email hash
        return valid, email_sha

    def get_hruid_for_userid(self, user_id):
        query = f"SELECT hr_uid FROM {self.database}.REGISTERED WHERE user_id = '{user_id}'"
        hr_uid, _ = self.execute_sql_query(query=query, commit=False)
        return hr_uid[0][0]

    def insert_or_update_user(self, email, first_name, last_name):
        registered, email_sha = self.is_email_registered(email)
        if not registered:
            query = f"""
            INSERT INTO {self.database}.REGISTERED (user_id, email, first_name, last_name, sessions)
            VALUES ('{email_sha}', '{email}', '{first_name}', '{last_name}', 0)
            """
            self.execute_sql_query(query=query, commit=True)
        else:

            self.logger.info(f"User with email {email} is already registered.")
