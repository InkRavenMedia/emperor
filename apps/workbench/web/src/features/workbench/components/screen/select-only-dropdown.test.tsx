import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { SelectOnlyDropdown } from "@/features/workbench/components/screen/select-only-dropdown";
import { DialogShell } from "@/features/workbench/components/shared/dialog-shell";

const options = [
  { value: "linear", label: "Linear" },
  { value: "neuron", label: "Neuron" },
  { value: "attention", label: "Attention" },
];

describe("SelectOnlyDropdown", () => {
  it("dismisses the listbox before its containing dialog on Escape", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(
      <DialogShell ariaLabel="Configuration" onClose={onClose}>
        <SelectOnlyDropdown
          label="Model"
          value="linear"
          options={options}
          onChange={vi.fn()}
        />
      </DialogShell>,
    );

    const trigger = screen.getByRole("button", { name: "Model" });
    await user.click(trigger);
    expect(screen.getByRole("combobox", { name: "Search Model" })).toHaveFocus();

    await user.keyboard("{Escape}");

    expect(onClose).not.toHaveBeenCalled();
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
    expect(trigger).toHaveFocus();

    await user.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalledOnce();
  });

  it("only references popup descendants while they are mounted", async () => {
    const user = userEvent.setup();

    render(
      <SelectOnlyDropdown
        label="Model"
        value="linear"
        options={options}
        onChange={() => {}}
      />,
    );

    const control = screen.getByRole("button", { name: "Model" });
    expect(control).not.toHaveAttribute("aria-controls");
    expect(control).not.toHaveAttribute("aria-activedescendant");

    await user.click(control);
    const popup = screen.getByRole("listbox", { name: "Model options" });
    const search = screen.getByRole("combobox", { name: "Search Model" });
    expect(control).toHaveAttribute("aria-controls", popup.id);
    expect(document.getElementById(search.getAttribute("aria-activedescendant") ?? ""))
      .toBeInTheDocument();

    await user.keyboard("{Escape}");
    expect(popup).not.toBeInTheDocument();
    expect(control).not.toHaveAttribute("aria-controls");
    expect(control).not.toHaveAttribute("aria-activedescendant");
  });

  it("filters options with search and shows an empty result message", async () => {
    const user = userEvent.setup();

    render(
      <SelectOnlyDropdown
        label="Model"
        value="linear"
        options={[
          ...options,
          {
            value: "mixture_of_attention_heads",
            label: "Mixture of Attention Heads",
            description: "moah preset",
          },
        ]}
        onChange={() => {}}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Model" }));
    const search = screen.getByRole("combobox", { name: "Search Model" });

    await user.type(search, "moah");

    expect(
      screen.getByRole("option", { name: /mixture of attention heads/i }),
    ).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: "Linear" })).not.toBeInTheDocument();

    await user.clear(search);
    await user.type(search, "missing");

    expect(screen.getByText("No matching options")).toBeInTheDocument();
  });

  it("selects a filtered option by click", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();

    render(
      <SelectOnlyDropdown
        label="Model"
        value="linear"
        options={options}
        onChange={onChange}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Model" }));
    await user.type(screen.getByRole("combobox", { name: "Search Model" }), "neu");
    await user.click(screen.getByRole("option", { name: "Neuron" }));

    expect(onChange).toHaveBeenCalledWith("neuron");
  });

  it("opens a top-layer menu above a trigger near the viewport bottom", async () => {
    const user = userEvent.setup();
    const showPopover = vi.fn();
    const prototype = HTMLElement.prototype;
    const previous = Object.getOwnPropertyDescriptor(prototype, "showPopover");
    Object.defineProperty(prototype, "showPopover", { configurable: true, value: showPopover });
    try {
      render(
        <div style={{ overflow: "hidden" }}>
          <SelectOnlyDropdown label="Model" value="linear" options={options} onChange={vi.fn()} />
        </div>,
      );
      const trigger = screen.getByRole("button", { name: "Model" });
      const triggerTop = window.innerHeight - 80;
      vi.spyOn(trigger, "getBoundingClientRect").mockReturnValue(
        DOMRect.fromRect({ x: 100, y: triggerTop, width: 240, height: 40 }),
      );

      await user.click(trigger);

      const popup = screen.getByRole("listbox", { name: "Model options" })
        .closest("[popover='manual']");
      expect(popup).toBeInTheDocument();
      expect(showPopover.mock.contexts).toEqual([popup]);
      expect(popup).toHaveStyle({
        position: "fixed",
        top: `${triggerTop - 8}px`,
        left: "100px",
        width: "240px",
        transform: "translateY(-100%)",
      });
    } finally {
      if (previous) {
        Object.defineProperty(prototype, "showPopover", previous);
      } else {
        Reflect.deleteProperty(prototype, "showPopover");
      }
    }
  });

  it("keeps keyboard focus separate from the selected option", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();

    render(
      <SelectOnlyDropdown
        label="Model"
        value="linear"
        options={options}
        onChange={onChange}
      />,
    );

    const control = screen.getByRole("button", { name: "Model" });
    await user.click(control);

    const listbox = screen.getByRole("listbox", { name: "Model options" });
    const search = screen.getByRole("combobox", { name: "Search Model" });
    const selectedOption = within(listbox).getByRole("option", { name: "Linear" });
    const nextOption = within(listbox).getByRole("option", { name: "Neuron" });

    expect(selectedOption).toHaveAttribute("aria-selected", "true");
    expect(nextOption).toHaveAttribute("aria-selected", "false");

    await user.keyboard("{ArrowDown}");

    expect(search).toHaveAttribute("aria-activedescendant", nextOption.id);
    expect(search).toHaveFocus();
    expect(selectedOption).toHaveAttribute("aria-selected", "true");
    expect(nextOption).toHaveAttribute("aria-selected", "false");

    await user.keyboard("{Enter}");

    expect(onChange).toHaveBeenCalledWith("neuron");
    expect(screen.queryByRole("listbox", { name: "Model options" })).not.toBeInTheDocument();
  });

  it("clears the search query after closing and reopening", async () => {
    const user = userEvent.setup();

    render(
      <SelectOnlyDropdown
        label="Model"
        value="linear"
        options={options}
        onChange={() => {}}
      />,
    );

    const control = screen.getByRole("button", { name: "Model" });
    await user.click(control);
    await user.type(screen.getByRole("combobox", { name: "Search Model" }), "att");

    expect(screen.queryByRole("option", { name: "Linear" })).not.toBeInTheDocument();

    await user.keyboard("{Escape}");
    await user.click(control);

    expect(screen.getByRole("combobox", { name: "Search Model" })).toHaveValue("");
    expect(screen.getByRole("option", { name: "Linear" })).toBeInTheDocument();
  });

  it("renders disabled options without allowing selection", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();

    render(
      <SelectOnlyDropdown
        label="Model"
        value="linear"
        options={[
          { value: "linear", label: "Linear" },
          { value: "neuron", label: "Neuron", disabled: true },
        ]}
        onChange={onChange}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Model" }));

    const disabledOption = screen.getByRole("option", { name: "Neuron" });
    expect(disabledOption).toHaveAttribute("aria-disabled", "true");

    await user.click(disabledOption);

    expect(onChange).not.toHaveBeenCalled();
    expect(screen.getByRole("listbox", { name: "Model options" })).toBeInTheDocument();
  });

  it("honors an explicit disabled state when options are available", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();

    render(
      <SelectOnlyDropdown
        label="Model"
        value="linear"
        options={options}
        onChange={onChange}
        disabled
      />,
    );

    const control = screen.getByRole("button", { name: "Model" });
    expect(control).toBeDisabled();

    await user.click(control);

    expect(screen.queryByRole("listbox", { name: "Model options" })).not.toBeInTheDocument();
    expect(onChange).not.toHaveBeenCalled();
  });
});
