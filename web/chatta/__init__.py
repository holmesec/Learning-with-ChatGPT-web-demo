from flask import Flask, render_template, request, redirect
import os
import uuid
from dotenv import load_dotenv

load_dotenv()

UPLOAD_FOLDER = 'chatta/uploads'


def create_app():
    from chatta import api

    app = Flask(__name__)

    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    app.register_blueprint(api.bp)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/upload", methods=["POST"])
    def upload():
        if 'pdf' not in request.files:
            return 'Missing file', 400

        file = request.files['pdf']

        if file.filename == '':
            return 'Missing file', 400

        # Content could still differ from file extension, but we're not concerned
        if file.filename.rsplit('.', 1)[1].lower() != 'pdf':
            return 'File type not allowed', 400

        file_id = uuid.uuid4().hex
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], f'{file_id}.pdf'))
        return redirect(f'/chat/{file_id}')

    @app.route("/chat/<file_id>")
    def chat(file_id):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{file_id}.pdf')
        if not os.path.isfile(file_path):
            return redirect('/')

        return render_template("chat.html", file_id=file_id)

    @app.route("/ab-test/context-length")
    def ab_test_ctx_length():
        return render_template("ab_test_len.html")

    return app
