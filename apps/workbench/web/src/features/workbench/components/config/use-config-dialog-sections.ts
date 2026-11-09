import { useEffect, useMemo, useRef, useState } from "react";
import { flushSync } from "react-dom";
import { splitSectionSignature } from "@/lib/config";

// Manages which config sections are open in the full-config dialog, keeping the
// open set in sync with the section list and supporting jump-to-section (which
// opens the target, then scrolls to it).
export function useConfigDialogSections(
  sections: Array<{ key: string }>,
  autoOpenKey?: string,
  initialOpenKeys?: string[],
) {
  const sectionSignature = sections.map((section) => section.key).join("\u0000");
  const sectionKeys = useMemo(
    () => splitSectionSignature(sectionSignature),
    [sectionSignature],
  );
  const initialOpenKeySignature = (initialOpenKeys ?? sectionKeys).join("\u0000");
  const [openSectionKeys, setOpenSectionKeys] = useState(
    () => new Set(splitSectionSignature(initialOpenKeySignature)),
  );
  const previousAutoOpenKey = useRef(autoOpenKey);
  const areAllSectionsOpen =
    sectionKeys.length > 0 && sectionKeys.every((key) => openSectionKeys.has(key));
  const sectionRefs = useRef<Record<string, HTMLElement | null>>({});

  useEffect(() => {
    const initialOpen = splitSectionSignature(initialOpenKeySignature).filter(
      (key) => sectionKeys.includes(key),
    );
    const sectionKeySet = new Set(sectionKeys);
    const shouldResetOpenSections = previousAutoOpenKey.current !== autoOpenKey;
    previousAutoOpenKey.current = autoOpenKey;

    setOpenSectionKeys((current) => {
      if (shouldResetOpenSections) {
        return new Set(initialOpen);
      }
      const next = new Set<string>();
      for (const key of current) {
        if (sectionKeySet.has(key)) {
          next.add(key);
        }
      }
      for (const key of initialOpen) {
        next.add(key);
      }
      return next;
    });
  }, [sectionKeys, initialOpenKeySignature, autoOpenKey]);

  function toggleSection(key: string) {
    setOpenSectionKeys((current) => {
      const next = new Set(current);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  }

  function toggleAllSections() {
    setOpenSectionKeys(areAllSectionsOpen ? new Set<string>() : new Set(sectionKeys));
  }

  function setOpenSections(keys: string[]) {
    setOpenSectionKeys(new Set(keys));
  }

  function jumpToSection(key: string) {
    flushSync(() => {
      setOpenSectionKeys((current) => {
        if (current.has(key)) {
          return current;
        }
        const next = new Set(current);
        next.add(key);
        return next;
      });
    });
    const reduceMotion = window.matchMedia?.(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    sectionRefs.current[key]?.scrollIntoView({
      block: "start",
      behavior: reduceMotion ? "auto" : "smooth",
    });
  }

  return {
    openSectionKeys,
    areAllSectionsOpen,
    sectionRefs,
    toggleSection,
    toggleAllSections,
    setOpenSections,
    jumpToSection,
  };
}
