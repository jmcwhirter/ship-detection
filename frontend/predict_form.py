from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

class PredictForm(FlaskForm):
    image_url = StringField('Image', validators=[DataRequired()])
    threshold = StringField('Threshold', validators=[DataRequired()])
    submit = SubmitField('Predict')
