// Build the example workflows in the ComfyUI frontend itself and save them to the user's workflow folder.
//
// The frontend's own graph API makes the nodes, so every widget, link and growing input is exactly what
// the frontend would save; writing that JSON by hand is how a workflow ends up loading with shifted
// widget values. Paste it into the browser console of a ComfyUI that has this pack loaded; the files land
// in ComfyUI/user/default/workflows/, from where they are copied into the pack's workflows/.

(async () => {
  const app = window.app;
  const LG = window.LiteGraph;

  const HF = {
    instruct_distil: "https://huggingface.co/PedroMarinhoDev/HunyuanImage-3.0-Instruct-Distil-ComfyUI",
    instruct: "https://huggingface.co/PedroMarinhoDev/HunyuanImage-3.0-Instruct-ComfyUI",
    base: "https://huggingface.co/PedroMarinhoDev/HunyuanImage-3.0-Base-ComfyUI",
  };
  const NAMES = { instruct_distil: "Instruct-Distil", instruct: "Instruct", base: "Base" };
  const SETTINGS = {
    instruct_distil: "**8 steps, cfg 1.0**, euler / simple, and **guidance 2.5** on the Guidance node. " +
            "About 26 s for a 1024×1024 image on an RTX 4090.",
    instruct: "**50 steps, cfg 2.5**, euler / simple. About 5 min for a 1024×1024 image on an RTX 4090, " +
              "or about 1.5 min with Spectrum on.",
    base: "**50 steps, cfg 5.0**, euler / simple. About 5 min for a 1024×1024 image on an RTX 4090, " +
          "or about 1.5 min with Spectrum on.",
  };
  const CREDIT = "\n\n---\n\nCustom Nodes and Workflow developed by **Pedro Marinho** · " +
                 "[x.com/PedroMarinhoDev](https://x.com/PedroMarinhoDev)\n\n" +
                 "[github.com/PedroMarinhoDev/ComfyUI-HunyuanImage3](https://github.com/PedroMarinhoDev/ComfyUI-HunyuanImage3)";

  // a file name as a link to its page in the model's HuggingFace repo; plain text, since the note's
  // markdown renderer drops a link wrapped around `code`
  const file = (model, path) => `[${path.split("/").pop()}](${HF[model]}/blob/main/${path})`;

  function readme(model, img2img) {
    const name = `hunyuan_image_3_${model}`;
    const rows = [
      `| ${file(model, `${name}_w4a8.safetensors`)}<br>or ${file(model, `${name}_int8_convrot.safetensors`)}` +
        `<br>or ${file(model, `${name}_bf16.safetensors`)} | \`models/diffusion_models\` |`,
      `| ${file(model, "vae/hunyuan_image_3_vae_fp16.safetensors")} | \`models/vae\` |`,
    ];
    if (img2img) rows.push(`| ${file(model, `clip_vision/${name}_siglip2_so400m_naflex.safetensors`)} | \`models/clip_vision\` |`);
    if (model !== "base") rows.push(`| ${file(model, `${name}_cot_head.safetensors`)}<br>(only for Prompt Rewriting) | \`models/diffusion_models\` |`);
    const task = img2img ? "image editing (1 to 3 images)" : "text to image";
    let how = img2img
      ? "**How to use:** load your image, write the edit in **Image Encode** (\"make it winter\", \"turn it into " +
        "a watercolour\"). To combine images, connect up to three: a new socket appears on Image Encode as you " +
        "connect one, and the prompt can refer to \"the first image\", \"the second image\". The output keeps " +
        "the first image's shape at about 1 megapixel (Scale Image to Total Pixels → Get Image Size); for a " +
        "fixed size, wire a Resolutions node instead.\n\n**Tip:** use a different seed from the one that made " +
        "the input image, or the edit can come out overcooked."
      : "**How to use:** write your prompt in **Text Encode** and pick a size on **Resolutions**.";
    return `## HunyuanImage 3.0 ${NAMES[model]} · ${task}\n\n` +
      `**Model files** — from [HuggingFace](${HF[model]}), into these folders (one of the three model formats: ` +
      "W4A8 is the smallest and fastest):\n\n" +
      "| File | Folder |\n|---|---|\n" + rows.join("\n") + "\n\n" +
      `**Settings:** ${SETTINGS[model]}\n\n${how}\n\n` +
      "Needs a 24 GB GPU and about 50 GB of free system RAM; the model streams from RAM as it runs." + CREDIT;
  }

  const SPECTRUM_NOTE = "**Spectrum** — set **enabled** to *true* for faster generations, at the expense of " +
                        "some quality. It skips some of the model's steps and predicts them instead.";
  const REWRITE_NOTE = "**Prompt Rewriting** — set **enabled** to *true* to let the model rewrite your prompt " +
                       "into a detailed one before drawing (*think + rewrite* reasons about it first). Slow: " +
                       "about 1 s per word it writes, so a few minutes. The rewrite shows in the text preview.";

  function spec(model, img2img) {
    const distil = model === "instruct_distil";
    const rewriting = model !== "base";
    const n = {
      readme: { type: "MarkdownNote", title: "Read me", widgets: { text: readme(model, img2img) }, size: [660, 0] },
      loader: { type: "HunyuanImage3ModelLoader", widgets: { model: `hunyuan_image_3_${model}_w4a8.safetensors` } },
      vae: { type: "HunyuanImage3VAELoader", widgets: { vae_name: "hunyuan_image_3_vae_fp16.safetensors" } },
      spectrum: { type: "HunyuanImage3Spectrum", widgets: { enabled: false } },
      spectrum_note: { type: "MarkdownNote", title: "Spectrum", widgets: { text: SPECTRUM_NOTE }, size: [0, 110] },
      latent: { type: "HunyuanImage3EmptyLatent" },
      sampler: { type: "KSampler", widgets: { seed: 1234, control_after_generate: "fixed", steps: distil ? 8 : 50,
                                                cfg: { instruct_distil: 1.0, instruct: 2.5, base: 5.0 }[model],
                                                sampler_name: "euler", scheduler: "simple", denoise: 1.0 } },
      decode: { type: "VAEDecode" },
      save: { type: "SaveImage", widgets: { filename_prefix: `hunyuan_image_3_${model}${img2img ? "_edit" : ""}` },
              size: [520, 580] },
    };
    const encoder = img2img ? "HunyuanImage3ImageEncode" : "HunyuanImage3TextEncode";
    n.encode = { type: encoder, size: [420, 0], widgets: { prompt: img2img
      ? "Turn the photograph into a watercolour painting"
      : "A red fox asleep curled in tall grass at golden hour, telephoto bokeh" } };
    if (rewriting) {
      n.rewrite = { type: "HunyuanImage3PromptRewriting", widgets: { enabled: false } };
      n.rewrite_note = { type: "MarkdownNote", title: "Prompt Rewriting", widgets: { text: REWRITE_NOTE }, size: [0, 150] };
      n.preview = { type: "PreviewAny", title: "Rewritten prompt", size: [420, 160] };
    }
    if (distil) n.guidance = { type: "HunyuanImage3Guidance" };
    if (img2img) {
      n.clip = { type: "CLIPVisionLoader", widgets: { clip_name: `hunyuan_image_3_${model}_siglip2_so400m_naflex.safetensors` } };
      n.image = { type: "LoadImage", widgets: { image: "hunyuan_image_3_conditioning.png" }, size: [340, 400] };
      n.scale = { type: "ImageScaleToTotalPixels", widgets: { upscale_method: "lanczos", megapixels: 1.0, resolution_steps: 16 } };
      n.size = { type: "GetImageSize" };
    } else {
      n.resolutions = { type: "HunyuanImage3Resolutions" };
    }

    const links = [
      ["loader", "model", "spectrum", "model"],
      ["spectrum", "model", "sampler", "model"],
      ["loader", "model", "encode", "model"],
      ["latent", "LATENT", "sampler", "latent_image"],
      ["encode", "negative", "sampler", "negative"],
      ["sampler", "LATENT", "decode", "samples"],
      ["vae", "VAE", "decode", "vae"],
      ["decode", "IMAGE", "save", "images"],
    ];
    if (distil) links.push(["encode", "positive", "guidance", "conditioning"], ["guidance", "CONDITIONING", "sampler", "positive"]);
    else links.push(["encode", "positive", "sampler", "positive"]);
    if (rewriting) links.push(["rewrite", "prompt_rewriting", "encode", "prompt_rewriting"],
                              ["encode", "rewritten_prompt", "preview", "source"]);
    const sizer = img2img ? "size" : "resolutions";
    links.push([sizer, "width", "encode", "width"], [sizer, "height", "encode", "height"],
               [sizer, "width", "latent", "width"], [sizer, "height", "latent", "height"]);
    if (img2img) links.push(["vae", "VAE", "encode", "vae"], ["clip", "CLIP_VISION", "encode", "clip_vision"],
                            ["image", "IMAGE", "encode", "images.image_1"], ["image", "IMAGE", "scale", "image"],
                            ["scale", "IMAGE", "size", "image"]);

    // left to right in the order the graph flows; each column stacks top to bottom
    const columns = img2img
      ? [["readme"], ["loader", "vae", "clip", "spectrum", "spectrum_note"], ["image", "scale", "size"],
         rewriting ? ["rewrite", "rewrite_note", "encode", "preview"] : ["encode"],
         distil ? ["guidance", "latent", "sampler"] : ["latent", "sampler"], ["decode", "save"]]
      : [["readme"], ["loader", "vae", "spectrum", "spectrum_note"],
         rewriting ? ["rewrite", "rewrite_note", "resolutions"] : ["resolutions"],
         rewriting ? ["encode", "preview"] : ["encode"],
         distil ? ["guidance", "latent", "sampler"] : ["latent", "sampler"], ["decode", "save"]];
    return { nodes: n, links, columns };
  }

  function build({ nodes, links, columns }) {
    app.graph.clear();
    const made = {};
    for (const [key, def] of Object.entries(nodes)) {
      const node = LG.createNode(def.type);
      if (!node) throw new Error(`unknown node type ${def.type}`);
      app.graph.add(node);
      if (def.title) node.title = def.title;
      for (const [name, value] of Object.entries(def.widgets || {})) {
        const widget = node.widgets.find(w => w.name === name);
        if (!widget) throw new Error(`${key} has no widget ${name}`);
        widget.value = value;
      }
      made[key] = node;
    }
    for (const [source, output, target, input] of links) {
      const from = made[source], to = made[target];
      const outputSlot = from.outputs.findIndex(o => o.name === output || o.type === output);
      const inputSlot = to.inputs.findIndex(i => i.name === input);
      if (outputSlot < 0 || inputSlot < 0) throw new Error(`cannot link ${source}.${output} -> ${target}.${input}`);
      from.connect(outputSlot, to, inputSlot);
    }
    // sizes: the node's own minimum, widened or heightened where the spec asks
    for (const [key, def] of Object.entries(nodes)) {
      const node = made[key];
      const minimum = node.computeSize();
      const [width, height] = def.size || [0, 0];
      node.size = [Math.max(minimum[0], width, def.type === "MarkdownNote" ? 360 : 0),
                   Math.max(minimum[1], height, key === "readme" ? 640 : 0)];
    }
    const TITLE = LG.NODE_TITLE_HEIGHT || 30, GAP_X = 70, GAP_Y = 40;
    let x = 0;
    for (const column of columns) {
      const width = Math.max(...column.map(key => made[key].size[0]));
      let y = 0;
      for (const key of column) {
        const node = made[key];
        if (nodes[key].type === "MarkdownNote" || key === "encode" || key === "preview") node.size[0] = width;
        node.pos = [x, y + TITLE];
        y += node.size[1] + TITLE + GAP_Y;
      }
      x += width + GAP_X;
    }
    app.graph.setDirtyCanvas(true, true);
    return app.graph.serialize();
  }

  const WORKFLOWS = {
    "hunyuan_image_3_instruct_distil_txt2img.json": ["instruct_distil", false],
    "hunyuan_image_3_instruct_distil_img2img.json": ["instruct_distil", true],
    "hunyuan_image_3_instruct_txt2img.json": ["instruct", false],
    "hunyuan_image_3_instruct_img2img.json": ["instruct", true],
    "hunyuan_image_3_base_txt2img.json": ["base", false],
  };
  const saved = [];
  for (const [file, [model, img2img]] of Object.entries(WORKFLOWS)) {
    const workflow = build(spec(model, img2img));
    workflow.extra = { ...(workflow.extra || {}), ds: { scale: 0.6, offset: [60, 60] } };
    const response = await fetch(`/api/userdata/${encodeURIComponent("workflows/" + file)}?overwrite=true`,
                                 { method: "POST", body: JSON.stringify(workflow, null, 1) });
    saved.push(`${file}: ${response.status}, ${workflow.nodes.length} nodes, ${workflow.links.length} links`);
  }
  return saved;
})();
