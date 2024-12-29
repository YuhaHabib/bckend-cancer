const Hapi = require('@hapi/hapi');
const tf = require('@tensorflow/tfjs-node');
const Boom = require('@hapi/boom');
const uuid = require('uuid');
const { Firestore } = require('@google-cloud/firestore');
require("dotenv").config();

const init = async () => {
    const server = Hapi.server({
        port: parseInt(process.env.PORT),
        host: process.env.HOST,
        routes: {
            cors: {
              origin: ['*'],
              additionalHeaders: ['authorization', 'content-type']
            },
        },
    });

    const firestore = new Firestore();

 
    const modelUrl = process.env.MODEL_URL;

    let model;
    try {

        model = await tf.loadGraphModel(modelUrl);
        console.log('Model loaded successfully');
    } catch (err) {
        console.error('Error loading model:', err);
    }

    server.route({
        method: 'OPTIONS',
        path: '/{any*}',
        handler: (request, h) => {
            return h
                .response()
                .header('Access-Control-Allow-Origin', 'https://angular-stacker-446203-s0.et.r.appspot.com')
                .header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
                .header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
        },
    });

    server.route({
        method: 'POST',
        path: '/predict',
        options: {
            payload: {
                output: 'stream',
                parse: true,
                multipart: true,
                maxBytes: 1000000,
            },
        },
        handler: async (request, h) => {
            try {
                const { payload } = request;

                if (!payload.image) {
                    return h.response({
                        status: "fail",
                        message: "Image is required",
                     }).code(400)
                }

                const file = payload.image;
                const chunks = [];

                for await (const chunk of file) {
                    chunks.push(chunk);
                }

                const buffer = Buffer.concat(chunks);
                
                const imageTensor = tf.node
                .decodeJpeg(buffer) 
                .resizeNearestNeighbor([224, 224])
                .expandDims()
                .toFloat()
                
                const prediction = model.predict(imageTensor);
                const score = await prediction.data()
                
                const confidientScore = Math.max(...score) * 100
                let result = 0
                
                if (confidientScore > 0.5) {
                    result = 1
                } else  {
                    result = 0            
                }
                
                // Map prediction to response
                const resultMap = {
                    0: {
                        result: 'Non-cancer',
                        suggestion: 'Penyakit kanker tidak terdeteksi.',
                    },
                    1: {
                        result: 'Cancer',
                        suggestion: 'Segera periksa ke dokter!',
                    },
                };
                
                const response = resultMap[result] || {
                    result: 'Unknown',
                    suggestion: 'Hasil tidak dapat diinterpretasikan.',
                };
                
                // Create prediction result
                const predictionResult = {
                    id: uuid.v4(),
                    result: response.result,
                    suggestion: response.suggestion,
                    createdAt: new Date().toISOString(),
                };
                
                // Store result in Firestore
                await firestore.collection('predictions').doc(predictionResult.id).set(predictionResult);
                
                return h.response({
                    status: 'success',
                    message: 'Model is predicted successfully',
                    data: predictionResult,
                }).code(201);
            } catch (error) {
                if (Boom.isBoom(error, 413)) {
                    return h.response({
                        status: "fail",
                        message: "Payload content length greater than maximum allowed: 1000000",
                     }).code(413)
                }

                return h.response({
                    status: "fail",
                    message: "Terjadi kesalahan dalam melakukan prediksi",
                 }).code(400)
            }
        },
    });

    // Grab history
    server.route({
        method: 'GET',
        path: '/predict/histories',
        handler: async (request, h) => {
            try {
                const predictionsSnapshot = await firestore.collection('predictions').get();

                if (predictionsSnapshot.empty) {
                    return h.response({
                        status: 'success',
                        data: [],
                    }).code(200);
                }

                const data = predictionsSnapshot.docs.map((doc) => {
                    const history = doc.data(); 
                    return {
                        id: doc.id,
                        history: {
                            ...history,
                        },
                    };
                });

                return h.response({
                    status: 'success',
                    data,
                }).code(200);
            } catch (error) {
                console.error(error);
                throw Boom.internal('Terjadi kesalahan saat mengambil data history');
            }
        },
    });

    server.ext('onPreResponse', (request, h) => {
        const response = request.response;
        if (response.isBoom && response.output.statusCode === 413) {
            return h.response({
                status: "fail",
                message: "Payload content length greater than maximum allowed: 1000000",
             }).code(413)
        }
        return h.continue;
    });

    await server.start();
    console.log(`Server running on ${server.info.uri}`);
};

process.on('unhandledRejection', (err) => {
    console.error(err);
    process.exit(1);
});

init();