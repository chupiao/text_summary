import os
from flask import Flask, request
import json
from summary.summary import GetSummary
app = Flask(__name__)

@app.route('/api/GetSummary/', methods=['GET', 'POST'])
def GetSumary():
	data = request.data
	data = data.decode(encoding="utf-8")
	content = json.loads(data)
	text = content['text']
	title = content['title']
	textsummary = GetSummary()
	textsummary.SetText(title, text)
	get_summary = textsummary.GetSummarization()
	summary=list()
	summary.append(get_summary)
	return json.dumps(summary)

@app.route('/')
def index():
	# 直接返回静态文件
	return app.send_static_file("index.html")
if __name__ == '__main__':
	# app.run(debug=True)
	port = int(os.environ.get("PORT", "5000"))
	app.run(host='127.0.0.1', port=port, debug=True)