<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Web Mobile dengan Flask, Bootstrap, dan OpenCV</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-5">
        <h1>Web dengan Tampilan Frame OpenCV</h1>
        <p>Tampilan kamera dari OpenCV:</p>
        <img src="{{ url_for('video_feed_face') }}" alt="Tampilan Kamera" width="340" height="240">
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.min.js"></script>
</body>
</html>
