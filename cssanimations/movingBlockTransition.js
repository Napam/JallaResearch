(() => {
  const container = createCard();

  const header = document.createElement("h3");
  header.textContent = "Moving block using transition";
  container.appendChild(header);

  container.style = css`
    display: flex;
    flex-direction: column;
    gap: 22px;
  `;

  const block = document.createElement("div");
  const blockSize = 40;
  block.style = css`
    width: 40px;
    height: 40px;
    background-color: #fff;
    transition: margin 0.5s;
  `;

  container.appendChild(block);

  let isRight = false;
  const button = document.createElement("button");
  button.textContent = "Animate";

  button.onclick = () => {
    if (isRight) {
      block.style.marginRight = `calc(100% - ${blockSize}px)`;
      block.style.marginLeft = "0px";
    } else {
      block.style.marginRight = "0px";
      block.style.marginLeft = `calc(100% - ${blockSize}px)`;
    }

    isRight = !isRight;
  };
  container.appendChild(button);
})();
