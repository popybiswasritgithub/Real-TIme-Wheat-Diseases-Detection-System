# # full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'images.jpg')
# #
#         # construct the argument parse and parse the arguments
#         model_path = "model.h5"
#         model = load_model(model_path)
#         lb = pickle.loads(open("label", "rb").read())
#
#         mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
#         Q = deque(maxlen=128)
#         input = full_filename
#
#         # noinspection PyArgumentList
#         vs = cv2.VideoCapture(input)
#
#         (W, H) = (None, None)
#
#         while True:
#             (grabbed, frame) = vs.read()
#
#             if not grabbed:
#                 break
#
#             if W is None or H is None:
#                 (H, W) = frame.shape[:2]
#
#             output = frame.copy()
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame = cv2.resize(frame, (224, 224)).astype("float32")
#             frame -= mean
#
#             preds = model.predict(np.expand_dims(frame, axis=0))[0]
#             Q.append(preds)
#
#             results = np.array(Q).mean(axis=0)
#             i = np.argmax(results)
#             label = lb.classes_[i]
#             window_name = 'Prediction'
#             text = "PREDICTION: {}".format(label.upper())
#
#             # putText(image_src, text, org, font, font_scale, color, thickness, line_type )
#             # cv2.putText(output, text, (50, 5), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 2)
#
#             # cv2_imshow(output)
#             # key = cv2.waitKey(10) & 0xFF
#
#         vs.release()
#         print(text)
# #
