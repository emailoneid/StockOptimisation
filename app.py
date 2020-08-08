from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client.dbsparta


## HTML 화면 보여주기
@app.route('/')
def homework():
    return render_template('index.html')


# 주문하기(POST) API
@app.route('/order', methods=['POST'])
def save_order():
    name = request.form['namedata']
    order = request.form['orderdata']
    address = request.form['addressdata']
    phone = request.form['phonedata']

    table = {'name': name, 'order': order, 'address': address, 'phone': phone}
    db.shopping.insert_one(table)

    return jsonify({'result': 'success'})


# 주문 목록보기(Read) API
@app.route('/order', methods=['GET'])
def view_orders():

    result = list(db.shopping.find({}, {'_id': 0}))

# 여길 채워나가세요!
    return jsonify({'result': 'success', 'shopping': result})


if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)