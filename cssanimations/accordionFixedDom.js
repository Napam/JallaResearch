(() => {
  const container = createCard();

  const header = document.createElement("h3");
  header.textContent = "Accordion with fixed DOM";
  container.appendChild(header);

  const body = document.createElement("ul");
  body.style = css`
    display: flex;
    flex-direction: column;
    background-color: #fff;
    width: 100%;
  `;
  container.appendChild(body);

  function createItem() {
    const item = document.createElement("li");
    item.style = css`
      width: 100%;
    `;

    const button = document.createElement("button");
    button.textContent = "Toggle";
    button.style = css`
      width: 100%;
      background-color: #f0f0f0;
      cursor: pointer;
    `;
    item.appendChild(button);

    let open = false;
    const content = document.createElement("div");
    content.style = css`
      transition: height 0.2s;
      height: 0;
      background-color: #777;
      overflow: hidden;
    `;

    const innerContent = document.createElement("div");
    innerContent.style = css`
      padding: 8px;
    `;

    innerContent.textContent =
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.";

    content.appendChild(innerContent);

    item.appendChild(content);

    button.onclick = () => {
      if (open) {
        content.style.height = "0";
      } else {
        content.style.height = content.scrollHeight + "px";
      }

      open = !open;
    };

    return item;
  }

  const elements = [
    createItem(),
    createItem(),
    createItem(),
    createItem(),
    createItem(),
  ];

  elements.forEach((element, i) => {
    body.appendChild(element);
  });
})();
