import os


def read_schema(schema_path):
    '''
    Read the .schema file
    '''
    # TODO
    with open(schema_path, 'r') as f:
        schema = f.read()
    return schema

def extract_sql_query(response):
    '''
    Extract the SQL query from the model's response
    '''
    # TODO
    start_token = "SELECT"
    end_token = ";"
    start_index = response.find(start_token)
    end_index = response.find(end_token, start_index) + len(end_token)
    if start_index != -1 and end_index != -1:
        return response[start_index:end_index]
    else:
        return ""
    

def save_logs(output_path, sql_em, record_em, record_f1, error_msgs):
    '''
    Save the logs of the experiment to files.
    You can change the format as needed.
    '''
    with open(output_path, "w") as f:
        f.write(f"SQL EM: {sql_em}\nRecord EM: {record_em}\nRecord F1: {record_f1}\nModel Error Messages: {error_msgs}\n")