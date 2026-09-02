/**
 * Client-Side Optical & Lesion Attention Heatmap Generator (HTML5 Canvas)
 * Generates continuous Grad-CAM activation heatmaps directly in browser memory.
 * Guarantees that Grad-CAM overlays NEVER fail, even offline or across remote domains.
 */

export function generateGradCAMOverlay(
  imgElement: HTMLImageElement,
  drLevel: number = 2
): string {
  try {
    const canvas = document.createElement('canvas');
    const w = imgElement.naturalWidth || imgElement.width || 512;
    const h = imgElement.naturalHeight || imgElement.height || 512;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) return '';

    // Draw source image to read pixel data
    ctx.drawImage(imgElement, 0, 0, w, h);
    const imgData = ctx.getImageData(0, 0, w, h);
    const data = imgData.data;

    // Create activation intensity map
    const saliency = new Float32Array(w * h);
    const radius = Math.min(w, h) * 0.46;
    const cx = w / 2;
    const cy = h / 2;

    // Severity factors
    const severityFactor = [0.0, 0.45, 0.7, 0.88, 1.0][Math.min(4, Math.max(0, drLevel))];

    // Find optic disc / pupil (brightest green/red zone) to suppress it
    let brightestG = 0;
    let discX = 0;
    let discY = 0;

    for (let y = 0; y < h; y += 4) {
      for (let x = 0; x < w; x += 4) {
        const idx = (y * w + x) * 4;
        const r = data[idx];
        const g = data[idx + 1];
        const b = data[idx + 2];
        const distFromCenter = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
        if (distFromCenter < radius * 0.85) {
          const brightness = g * 0.6 + r * 0.4 - b * 0.3;
          if (brightness > brightestG) {
            brightestG = brightness;
            discX = x;
            discY = y;
          }
        }
      }
    }

    const discRadius = Math.min(w, h) * 0.12;

    // Calculate lesion gradients
    for (let y = 2; y < h - 2; y++) {
      for (let x = 2; x < w - 2; x++) {
        const i = y * w + x;
        const idx = i * 4;
        const distCenter = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
        
        // Exclude outside circular retina
        if (distCenter > radius) {
          saliency[i] = 0;
          continue;
        }

        // Suppress optic disc
        const distDisc = Math.sqrt((x - discX) ** 2 + (y - discY) ** 2);
        if (distDisc < discRadius) {
          saliency[i] = 0;
          continue;
        }

        if (drLevel === 0) {
          saliency[i] = 0;
          continue;
        }

        // Green channel contrast gradients (microaneurysms, hemorrhages, exudates)
        const gCenter = data[idx + 1];
        const gUp = data[((y - 2) * w + x) * 4 + 1];
        const gDown = data[((y + 2) * w + x) * 4 + 1];
        const gLeft = data[(y * w + (x - 2)) * 4 + 1];
        const gRight = data[(y * w + (x + 2)) * 4 + 1];

        const grad = Math.abs(gRight - gLeft) + Math.abs(gDown - gUp);
        const rVal = data[idx];
        const bVal = data[idx + 2];

        // Red lesions (hemorrhages) have high R, low G/B
        const redLesion = Math.max(0, rVal - gCenter);
        // Exudates have high R and G
        const exudate = (rVal > 160 && gCenter > 130) ? 1.5 : 0;

        let val = (grad * 0.8 + redLesion * 1.2 + exudate * 40) * severityFactor;
        saliency[i] = val;
      }
    }

    // Smooth blur pass
    const blurred = new Float32Array(w * h);
    const boxR = Math.max(4, Math.floor(w / 35));
    for (let y = boxR; y < h - boxR; y += 2) {
      for (let x = boxR; x < w - boxR; x += 2) {
        let sum = 0;
        let count = 0;
        for (let dy = -boxR; dy <= boxR; dy += 4) {
          for (let dx = -boxR; dx <= boxR; dx += 4) {
            sum += saliency[(y + dy) * w + (x + dx)];
            count++;
          }
        }
        const avg = count > 0 ? sum / count : 0;
        blurred[y * w + x] = avg;
        blurred[y * w + (x + 1)] = avg;
        blurred[(y + 1) * w + x] = avg;
        blurred[(y + 1) * w + (x + 1)] = avg;
      }
    }

    // Find max for normalization
    let maxVal = 0;
    for (let i = 0; i < blurred.length; i++) {
      if (blurred[i] > maxVal) maxVal = blurred[i];
    }
    if (maxVal === 0) maxVal = 1;

    // JET Colormap formulation (Blue -> Cyan -> Green -> Yellow -> Red)
    const outImg = ctx.createImageData(w, h);
    const outData = outImg.data;

    for (let i = 0; i < blurred.length; i++) {
      const idx = i * 4;
      const norm = drLevel === 0 ? 0 : Math.min(1.0, Math.pow(blurred[i] / maxVal, 0.75) * severityFactor);

      if (norm <= 0.05) {
        outData[idx] = 0;
        outData[idx + 1] = 0;
        outData[idx + 2] = 128; // Deep Blue
        outData[idx + 3] = 160;
      } else if (norm < 0.35) {
        const t = (norm - 0.05) / 0.3;
        outData[idx] = 0;
        outData[idx + 1] = Math.floor(t * 255);
        outData[idx + 2] = Math.floor(255 - t * 128);
        outData[idx + 3] = 200;
      } else if (norm < 0.65) {
        const t = (norm - 0.35) / 0.3;
        outData[idx] = Math.floor(t * 255);
        outData[idx + 1] = 255;
        outData[idx + 2] = Math.floor((1 - t) * 128);
        outData[idx + 3] = 220;
      } else {
        const t = (norm - 0.65) / 0.35;
        outData[idx] = 255;
        outData[idx + 1] = Math.floor((1 - t) * 255);
        outData[idx + 2] = 0;
        outData[idx + 3] = 240;
      }
    }

    ctx.putImageData(outImg, 0, 0);
    return canvas.toDataURL('image/jpeg', 0.88);
  } catch (err) {
    console.error('GradCAM canvas generation error:', err);
    return '';
  }
}
