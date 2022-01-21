const { expect } = require("@jest/globals")
const dateutils = require('./dateutils')

daysIn = {
  january2022: {
    days: 31,
    mondays: 5,
    tuesdays: 4,
    wednesdays: 4,
    thursdays: 4,
    fridays: 4,
    saturdays: 5,
    sundays: 5
  },
  february2022: {
    days: 28,
    mondays: 4,
    tuesdays: 4,
    wednesdays: 4,
    thursdays: 4,
    fridays: 4,
    saturdays: 4,
    sundays: 4
  },
  march2022: {
    days: 31,
    mondays: 4,
    tuesdays: 5,
    wednesdays: 5,
    thursdays: 5,
    fridays: 4,
    saturdays: 4,
    sundays: 4
  },
  april2022: {
    days: 30,
    mondays: 4,
    tuesdays: 4,
    wednesdays: 4,
    thursdays: 4,
    fridays: 5,
    saturdays: 5,
    sundays: 4
  }
}

test('countDays for January 2022 is correct', () => {
  from = new Date(2022, 0, 1)
  to = new Date(2022, 0, 31)
  expect(dateutils.countDays(from, to)).toEqual(daysIn.january2022)
})
test('countDays for February 2022 is correct', () => {
  from = new Date(2022, 1, 1)
  to = new Date(2022, 1, 28)
  expect(dateutils.countDays(from, to)).toEqual(daysIn.february2022)
})
test('countDays for March 2022 is correct', () => {
  from = new Date(2022, 2, 1)
  to = new Date(2022, 2, 31)
  expect(dateutils.countDays(from, to)).toEqual(daysIn.march2022)
})
test('countDays for April 2022 is correct', () => {
  from = new Date(2022, 3, 1)
  to = new Date(2022, 3, 30)
  expect(dateutils.countDays(from, to)).toEqual(daysIn.april2022)
})
test('countDays for Jan 1 2022 to April 31 2022 is correct', () => {
  from = new Date(2022, 0, 1)
  to = new Date(2022, 3, 30)
  expect(dateutils.countDays(from, to)).toEqual(dateutils.aggregate(Object.values(daysIn)))
})