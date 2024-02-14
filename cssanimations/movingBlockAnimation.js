(() => {
  const container = createCard();

  const header = document.createElement("h3");
  header.textContent = "Moving block using animation";
  container.appendChild(header);

  container.style = css`
    display: flex;
    flex-direction: column;
    gap: 22px;
  `;

  const keyframes = document.createElement("style");
  keyframes.textContent = css`
    @keyframes move-right {
      0% {
        margin-left: 0px;
        margin-right: calc(100% - 40px);
      }
      100% {
        margin-left: calc(100% - 40px);
        margin-right: 0px;
      }
    }

    @keyframes move-left {
      0% {
        margin-right: 0px;
        margin-left: calc(100% - 40px);
      }
      100% {
        margin-right: calc(100% - 40px);
        margin-left: 0px;
      }
    }
  `;
  document.head.appendChild(keyframes);

  const block = document.createElement("div");
  block.style = css`
    width: 40px;
    height: 40px;
    background-color: #fff;
  `;
  container.appendChild(block);

  let isRight = false;
  const button = document.createElement("button");
  button.textContent = "Animate";

  button.onclick = () => {
    if (isRight) {
      block.style.animation = "move-left 0.5s forwards";
    } else {
      block.style.animation = "move-right 0.5s forwards";
    }

    isRight = !isRight;
  };
  container.appendChild(button);
})();
