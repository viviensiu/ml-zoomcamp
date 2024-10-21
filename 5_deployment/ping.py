from flask import Flask

app = Flask('ping')

# the app.route decorator includes additional functionalities to the ping() function.
# app.route(url rule, method)
# URL rule: the url after the main index. Specify '/' for main index
# the GET method is used for retrieving info from backend.
@app.route('/ping', methods=['GET'])
def ping():
    return "PONG"

# __main__ function contains the running sequence 
# Executed first when calling a python script
if __name__ == "__main__":
    # 0.0.0.0 indicates to listen on any available network interface,
    # usually 127.0.0.1 (localhost) is used. 
    app.run(debug=True, host='0.0.0.0', port=9696)