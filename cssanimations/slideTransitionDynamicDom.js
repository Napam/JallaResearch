(() => {
  const container = createCard();
  container.style = css`
    display: flex;
    flex-direction: column;
  `;

  const header = document.createElement("h3");
  header.textContent = "Fade + slide in / out with dynamic DOM";
  container.appendChild(header);

  const innerDiv = document.createElement("div");
  innerDiv.style = css`
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    height: 100%;
    width: 100%;
  `;
  container.appendChild(innerDiv);


  const button = document.createElement("button");
  button.textContent = "Bring in";
  container.appendChild(button);

  const box = document.createElement("div");
  box.style = css`
    background-color: #fff;
    width: 100px;
    height: 100px;
    transition:
      opacity 0.2s,
      transform 0.2s;
  `;

  let isShowing = true;
  innerDiv.appendChild(box);

  button.addEventListener("click", () => {
    if (isShowing) {
      innerDiv.addEventListener("transitionend", () => {
        innerDiv.removeEventListener("transitionend");
        innerDiv.removeChild(box);
      });

      box.style.opacity = "0";
      box.style.transform = "translateX(110%)";

      button.textContent = "Bring in";
    } else {
      box.style.opacity = "0";
      box.style.transform = "translateX(-110%)";
      innerDiv.appendChild(box);

      // Trigger reflow
      box.offsetWidth;

      box.style.opacity = "1";
      box.style.transform = "translateX(0)";

      button.textContent = "Take out";
    }

    isShowing = !isShowing;
  });
})();
