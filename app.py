from flask import Flask,render_template,request
import pickle
import numpy as np
app=Flask(__name__)
model=pickle.load(open('Iris Data/model.pkl','rb'))

@app.route('/',methods=['POST','GET'])
def main():
    return render_template('index.html')

@app.route('/index',methods=['POST','GET'])
def home():
    slenth=request.args.get('slenth')
    swidth=request.args.get('swidth')
    plenth=request.args.get('plenth')
    pwidth=request.args.get('pwidth')
    a=np.array([[slenth,swidth,plenth,pwidth]])
    predicta=model.predict(a)
    print(predicta)
    if predicta==0:
        return render_template('index.html',name='Iris-setosa')
    elif predicta==1:
        return render_template('index.html',name='Iris-versicolor')
    elif predicta==2:
        return render_template('index.html',name='Iris-virginica')
        
    
        
        

if __name__=='__main__':    
    app.run(debug=True,port=6788)