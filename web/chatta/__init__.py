from flask import Flask, render_template, request, redirect
import os
from hashlib import sha256
import threading
from chatta.utils import process_pdf, UPLOAD_FOLDER


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

        id = sha256(file.read()).hexdigest()
        file.seek(0)

        embeddings_path = os.path.join(
            app.config['UPLOAD_FOLDER'], 'embeddings', f'{id}.pkl')

        # Check if embeddings already exists for the provided PDF
        if not os.path.isfile(embeddings_path):
            file.save(os.path.join(
                app.config['UPLOAD_FOLDER'], 'pdf', f'{id}.pdf'))
            thread = threading.Thread(
                target=process_pdf, kwargs={'id': id})
            thread.start()

        return redirect(f'/chat/{id}')

    @app.route("/chat/<id>")
    def chat(id):
        pdf_path = os.path.join(
            app.config['UPLOAD_FOLDER'], 'pdf', f'{id}.pdf')
        if not os.path.isfile(pdf_path):
            return redirect('/')
        return render_template("chat.html", id=id)

    @app.route("/ab-test/context-length")
    def ab_test_ctx_length():
        return render_template("ab_test_len.html")

    @app.route("/ab-test/context/<name>")
    def ab_test_ctx(name):
        if name not in ['2450', 'history']:
            return "A/B test not found!", 404
        return render_template("ab_test_ctx.html", name=name)

    return app
