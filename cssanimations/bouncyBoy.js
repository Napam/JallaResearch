(() => {
  const container = createCard();

  const header = document.createElement("h3");
  header.textContent = "Bouncy boy";
  container.appendChild(header);

  container.style = css`
    display: flex;
    flex-direction: column;
    gap: 22px;
  `;

  const innerDiv = document.createElement("div");
  innerDiv.style = css`
    display: flex;
    height: 140px;
    width: 100%;
    background-color: #777;
  `;
  container.appendChild(innerDiv);

  const keyframes = document.createElement("style");
  keyframes.textContent = css`
    @keyframes bounce-around {
      0% {
        transform: translate(0, 0);
      }
      25% {
        transform: translate(60px, 80px);
      }
      50% {
        transform: translate(100px, 0);
      }
      75% {
        transform: translate(60px, 80px);
      }
      100% {
        transform: translate(0, 0);
      }
    }
  `;
  document.head.appendChild(keyframes);

  const ball = document.createElement("div");
  ball.style = css`
    width: 40px;
    height: 40px;
    border-radius: 100%;
    background-color: #fff;
  `;
  innerDiv.appendChild(ball);

  let animating = true;
  const button = document.createElement("button");
  button.textContent = "Animate";

  ball.style.animation = "bounce-around 2s";
  button.onclick = () => {
    if (animating) {
    } else {
      ball.style.animation =
        "bounce-around 0.5s cubic-bezier(.1,.82,.25,1) infinite ";
    }

    animating = !animating;
  };
  container.appendChild(button);
})();
