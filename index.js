import * as wasm from "indicasr-web";

document.getElementById("upload_file").addEventListener(
  "change",
  function () {
    var reader = new FileReader();
    reader.onload = function () {
      let arrayBuffer = this.result;
      let array = new Uint8Array(arrayBuffer);
      let processed_data = wasm.run_preprocessor(array);
      inference(processed_data)
        .then((logprobs) => {
          console.log("done");
          const batch_size = processed_data.length;
          const vocab_arr = [
            "<unk>",
            "ा",
            "र",
            "ी",
            "▁",
            "े",
            "न",
            "ि",
            "त",
            "क",
            "्",
            "ल",
            "म",
            "स",
            "ं",
            "▁स",
            "ह",
            "ो",
            "ु",
            "द",
            "य",
            "प",
            "▁है",
            "▁के",
            "ग",
            "▁ब",
            "▁म",
            "व",
            "▁क",
            "▁में",
            "ट",
            "▁अ",
            "ज",
            "▁द",
            "▁प",
            "▁आ",
            "्र",
            "ू",
            "▁ज",
            "▁की",
            "▁र",
            "ध",
            "र्",
            "ों",
            "ख",
            "▁का",
            "्य",
            "च",
            "ए",
            "ब",
            "भ",
            "ने",
            "▁को",
            "▁से",
            "▁ल",
            "▁और",
            "▁प्र",
            "▁त",
            "▁कर",
            "▁व",
            "ता",
            "श",
            "▁कि",
            "▁ह",
            "▁न",
            "▁ग",
            "ना",
            "▁हो",
            "ै",
            "▁पर",
            "थ",
            "▁उ",
            "ड",
            "▁च",
            "िक",
            "ण",
            "ई",
            "▁हैं",
            "िया",
            "▁इस",
            "फ",
            "▁वि",
            "वा",
            "▁जा",
            "ष",
            "ित",
            "▁श",
            "ें",
            "▁ने",
            "ेश",
            "ते",
            "इ",
            "▁भी",
            "का",
            "▁एक",
            "्या",
            "▁हम",
            "▁सं",
            "िल",
            "ंग",
            "ड़",
            "छ",
            "क्ष",
            "ौ",
            "ठ",
            "़",
            "ॉ",
            "ओ",
            "ढ",
            "घ",
            "आ",
            "झ",
            "ऐ",
            "ँ",
            "ऊ",
            "उ",
            "ः",
            "औ",
            ",",
            "ऍ",
            "ॅ",
            "ॠ",
            "ऋ",
            "ऑ",
            "ञ",
            "ृ",
            "अ",
            "ङ",
            "b",
          ];

          const vocab_size = vocab_arr.length;
          const time_steps = logprobs.length / (vocab_size * batch_size);

          const text_batch = wasm.decode_logprobs(
            logprobs,
            [batch_size, time_steps, vocab_size],
            vocab_arr
          );
          console.log(text_batch);
          // console.log(document.getElementById("transcript"));

          document.getElementById("transcript").value = text_batch[0];
        })
        .catch((e) => {
          console.log("Error happened");
          console.log(e);
          console.log(e.stack);
        });
      // const session = ort.InferenceSession.create("./model.onnx");
    };
    reader.readAsArrayBuffer(this.files[0]);
  },
  false
);

async function inference(data) {
  const data_length = data[0][0].length;
  const i = data.length;
  const j = data[0].length;
  const k = data[0][0].length;
  let arr = [];
  for (let a = 0; a < i; a++) {
    for (let b = 0; b < j; b++) {
      for (let c = 0; c < k; c++) {
        arr.push(data[a][b][c]);
      }
    }
  }
  const audio_tensor = new ort.Tensor("float32", new Float32Array(arr), [
    i,
    j,
    k,
  ]);
  const audio_tensor_length = new ort.Tensor(
    "int64",
    new BigInt64Array([BigInt(data_length)])
  );
  const feeds = { audio_signal: audio_tensor, length: audio_tensor_length };
  const session = await ort.InferenceSession.create("./mymodel_updated.onnx");
  const results = await session.run(feeds);
  // console.log(results.logprobs.cpuData);
  return results.logprobs.cpuData;
}
