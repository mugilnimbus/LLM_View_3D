export function drawHeatmap(canvas, matrix) {
  const parentWidth = canvas.clientWidth || 420;
  const size = Math.max(280, Math.floor(parentWidth));
  const ratio = window.devicePixelRatio || 1;
  canvas.width = size * ratio;
  canvas.height = size * ratio;
  canvas.style.height = `${size}px`;

  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  context.clearRect(0, 0, size, size);

  if (!matrix?.length) {
    context.fillStyle = "#9cabaf";
    context.fillText("Run a prompt to see attention.", 20, 32);
    return;
  }

  const rows = matrix.length;
  const cols = matrix[0].length;
  const cellWidth = size / cols;
  const cellHeight = size / rows;

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      context.fillStyle = colorFor(matrix[row][col]);
      context.fillRect(col * cellWidth, row * cellHeight, Math.ceil(cellWidth), Math.ceil(cellHeight));
    }
  }

  context.strokeStyle = "rgba(244, 247, 248, 0.18)";
  context.lineWidth = 1;
  context.strokeRect(0.5, 0.5, size - 1, size - 1);
}

function colorFor(value) {
  const intensity = Math.max(0, Math.min(1, value * 4));
  const red = Math.round(20 + 222 * intensity);
  const green = Math.round(42 + 141 * Math.sqrt(intensity));
  const blue = Math.round(48 + 62 * (1 - intensity));
  return `rgb(${red}, ${green}, ${blue})`;
}
