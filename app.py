from flask import Flask,render_template, request,url_for
import pickle
import numpy as np

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))



@app.route("/")
def home():
    return render_template("base.html")

@app.route("/results", methods =["GET", "POST"])
def result():
    if request.method == "POST":
        
        # Read Values

        Fixedacidity = request.form.get("Fixed_acidity")
        Volatileacidity = request.form.get("Volatile_acidity")
        Citricacid=request.form.get("Citricacid")
        Residualsugar=request.form.get("Residual_sugar")
        Chlorides=request.form.get("Chlorides")
        Freesulfurdioxide=request.form.get("Free_sulfur_dioxide")
        Totalsulfurdioxide=request.form.get("Total_sulfur_dioxide")
        Density=request.form.get("Density")
        pH=request.form.get("pH")
        Sulphates=request.form.get("Sulphates")
        Alcohol=request.form.get("Alcohol")

        values=(Fixedacidity,Volatileacidity,Citricacid,Residualsugar,Chlorides,Freesulfurdioxide,Totalsulfurdioxide,Density,pH,Sulphates,Alcohol)
         
        # finalvalues=[np.array(values)]
        array_of_values=np.asarray(values)
        # Create functionality

        reshaped_array=array_of_values.reshape(1,-1)

        prediction=model.predict(reshaped_array)

        if prediction==1:
            output="The Quality of wine is Good"
        else:
            output="The Quality of wine is Bad"

        # Display result
        
        return render_template("test.html",result=output)


if __name__=="__main__":
    app.run(debug=True)